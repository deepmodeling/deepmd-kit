# SPDX-License-Identifier: LGPL-3.0-or-later
"""
HybridMuon optimizer for DeePMD-kit PyTorch backend.

HybridMuon is a hybrid optimizer that automatically combines Muon and Adam.
Routing is controlled by parameter dimensionality, parameter names, and
``muon_mode``:

- Parameters whose final effective name segment is ``b``, contains ``bias``,
  or starts with ``adam_`` (case-insensitive): Adam.
- Parameters whose final effective name segment starts with ``adamw_``
  (case-insensitive): Adam with decoupled weight decay (AdamW-style).
  The final effective segment means the last non-numeric segment in the full
  parameter path (split by ``"."``), so trailing ParameterList indices are
  ignored.
- 1D parameters (biases, norms, scales): Adam (no weight decay).
- ``muon_mode="2d"``:
  - Matrix parameters with effective rank 2 (after dropping singleton dims)
    use Muon.
  - Effective rank >2 parameters use Adam with decoupled weight decay fallback.
- ``muon_mode="flat"``:
  - >=2D matrix parameters use flattened matrix-view routing:
    ``(rows, cols) = (prod(effective_shape[:-1]), effective_shape[-1])``.
- ``muon_mode="slice"``:
  - Effective rank 2 matrix parameters: same as ``"2d"``.
  - Effective rank >=3 matrix parameters: treat leading axes as batch and apply Muon
    independently on each ``(..., m, n)`` slice (no cross-slice mixing).
  - Routing shape is computed on effective shape (singleton dims removed).

This is different from PyTorch's torch.optim.Muon, which ONLY supports 2D parameters
and requires manual configuration of AdamW for 1D parameters. HybridMuon provides
automatic routing based on parameter dimensionality.

Algorithm
---------
For Muon-routed parameters, the update is:

    1. Momentum update (Nesterov):
       m_t = beta * m_{t-1} + (1 - beta) * g_t
       update = beta * m_t + (1 - beta) * g_t

    2. Orthogonalization:
       - Standard path (hybrid two-stage Newton-Schulz, DeepSeek-V4 style):
         X_0 = G / ||G||_F
         A_k = X_k @ X_k^T
         X_{k+1} = a*X_k + (b*A_k + c*A_k^2) @ X_k
         * Fast stage: ``NS_STEPS_FAST`` iters with ``NS_COEFF_FAST``
           (3.4445, -4.7750, 2.0315) — drives singular values near 1.
         * Polish stage: ``NS_STEPS_POLISH`` iters with ``NS_COEFF_POLISH``
           (2.0, -1.5, 0.5) — exact Newton iteration stabilizes sigma at 1.
       - Gram path (when ``enable_gram=True`` and the matrix is rectangular):
         X_0 = G / ||G||_F
         R_k = X_k @ X_k^T
         Z_k = b*R_k + c*R_k^2
         Q_k = Z_k + a*I                              [restart]
         Q_k = Q_{k-1} @ (Z_k + a*I)                 [accumulation]
         RZ_k = a*R_k + R_k @ Z_k
         R_{k+1} = a*RZ_k + Z_k @ RZ_k
         X_out = Q_last @ X_restart
         Uses float32 normalization followed by float16 iteration with
         five Polar-Express coefficient tuples (kept unchanged; the
         Polar-Express recipe already tight-calibrates sigma→1 on its final
         step and does not accept an extra Newton polish cleanly).

    3. Scaling: scale = coeff * sqrt(max(m, n))  [match-RMS mode]
                scale = sqrt(max(1, m/n))        [rectangular mode]

    4. Parameter update: theta -= lr * scale * orth(update)

For Adam-routed parameters, standard Adam moments are used.
AdamW behavior (decoupled weight decay) is applied only on >=2D Adam paths.

Dtype Behavior
--------------
- Standard Newton-Schulz path: bfloat16 iterations
- Gram Newton-Schulz path: float32 normalization + float16 iterations
- NS output directly applied to parameters after casting back to the input dtype
- Adam state (exp_avg, exp_avg_sq): always float32 for numerical stability
- Muon momentum buffer: follows gradient dtype (grad -> buffer -> update)
- Adam gradients: cast to float32 for update computation

References
----------
.. [1] Keller Jordan, "Muon: An optimizer for hidden layers in neural networks."
       https://kellerjordan.github.io/posts/muon/
       https://github.com/KellerJordan/Muon
.. [2] Moonshot team, "Muon is Scalable for LLM Training," arXiv:2502.16982, 2025.
       https://arxiv.org/abs/2502.16982
.. [3] Moonlight GitHub Repository.
       https://github.com/MoonshotAI/Moonlight
.. [4] Flash-Muon: Triton-accelerated symmetric matmul for Newton-Schulz.
       https://github.com/lintianyang/flash-muon (MIT License, Tianyang Lin)
.. [5] Magma: Momentum-Aligned Gradient Masking for Stable Optimizer Updates.
       arXiv:2602.15322, 2026.
       https://arxiv.org/abs/2602.15322
       Implements block-wise momentum-gradient alignment scoring with EMA smoothing
       and soft scaling for improved stability under heavy-tailed gradient noise.
       HybridMuon uses a stabilized variant (Magma-lite) with sigmoid range stretching
       and continuous soft scaling [0.1, 1.0] instead of Bernoulli masking, optimized
       for MLIP force-field training.
.. [6] Dao-AILab, "gram-newton-schulz."
       https://github.com/Dao-AILab/gram-newton-schulz
.. [7] DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token
       Context Intelligence," 2026. Section 2.4 introduces the two-stage
       hybrid Newton-Schulz (8 fast + 2 polish iterations) and uses
       gamma=0.18 for update-RMS rescaling with AdamW ε=1e-20.
"""

from __future__ import (
    annotations,
)

import math
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch._dynamo.config as _dynamo_config
from torch.optim.optimizer import (
    Optimizer,
)

DYNAMO_CACHE_SIZE_LIMIT = 64
_dynamo_config.cache_size_limit = max(
    int(_dynamo_config.cache_size_limit),
    DYNAMO_CACHE_SIZE_LIMIT,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

# ============================================================================
# Triton availability detection
# ============================================================================

# Two-stage probe:
#   1. ``import triton`` succeeds (package is installed).
#   2. ``triton.runtime.driver.active`` resolves to a usable backend driver.
#
# Stage 2 is required because ``@triton.autotune(...)`` eagerly calls
# ``driver.active.get_benchmarker()`` inside ``Autotuner.__init__``; on
# CPU-only / driver-less hosts this raises ``RuntimeError: 0 active drivers``
# at *module import time*, breaking non-training entry points.
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    try:
        # Touching ``driver.active`` forces the lazy proxy to initialize the
        # backend driver. ``get_current_target`` is the lightest public call
        # that exercises the same path as ``Autotuner.__init__``.
        triton.runtime.driver.active.get_current_target()
        TRITON_AVAILABLE = True
    except Exception:
        # No usable runtime driver (no CUDA/ROCm/XPU, or a mis-configured
        # one): fall back to the pure-PyTorch Newton-Schulz path.
        TRITON_AVAILABLE = False
except ImportError:
    pass

# ============================================================================
# Constants
# ============================================================================

# --- Newton-Schulz two-stage iteration schedule (DeepSeek-V4 §2.4) ---
# Fast stage drives singular values close to 1 rapidly; polish stage uses
# exact Newton iteration (a=2, b=-1.5, c=0.5) to stabilize sigma precisely at 1.
# Only the Standard / Flash NS paths use this schedule. The Gram (Polar-
# Express) path is a different orthogonalization recipe and keeps its own
# pre-calibrated 5-step schedule in POLAR_EXPRESS_COEFFICIENTS below.
NS_STEPS_FAST: int = 8
NS_STEPS_POLISH: int = 2
NS_COEFF_FAST: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
NS_COEFF_POLISH: tuple[float, float, float] = (2.0, -1.5, 0.5)

# --- Numerical stability epsilons ---
# NS_EPS: guards Frobenius-norm clamp so X_0 = G / ||G||_F stays finite.
#   Normal gradients satisfy ||G||_F >> 1e-7, so this never bites in practice.
# ADAM_EPS: Adam denominator ε in ``sqrt(v_hat) + ε``. DeepSeek-V4 uses
#   ε=1e-20 so the Adam update is driven by the second-moment estimate rather
#   than the floor ε. For MLIP training (SeZM) this matters for ``adam_``
#   parameters whose gradient scale spans many orders of magnitude across
#   training (e.g. ``adam_ffn_layer_scales`` starting at 1e-3 vs
#   ``adam_type_embedding`` at O(1)).
NS_EPS: float = 1e-7
ADAM_EPS: float = 1e-20
# Polar Express coefficients with the safety scaling used in the reference repo
_GRAM_NS_UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS: tuple[
    tuple[float, float, float], ...
] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
GRAM_NS_SAFETY_FACTOR: float = 1.05
POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = tuple(
    (
        a / GRAM_NS_SAFETY_FACTOR,
        b / GRAM_NS_SAFETY_FACTOR**3,
        c / GRAM_NS_SAFETY_FACTOR**5,
    )
    for a, b, c in _GRAM_NS_UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS
)
# Minimum matrix dimension for flash path to be beneficial.
# Below this threshold, triton kernel launch overhead dominates over compute,
# and cuBLAS (via torch.mm/addmm) is faster for small matrices.
FLASH_MIN_DIM: int = 1024
# Magma-lite constants (Muon path update damping only)
MAGMA_TAU: float = 2.0
MAGMA_EMA_DECAY: float = 0.9
MAGMA_MIN_SCALE: float = 0.1
MAGMA_EPS: float = 1e-12
MAGMA_SIGMOID_MIN: float = 1.0 / (1.0 + math.exp(1.0 / MAGMA_TAU))
MAGMA_SIGMOID_MAX: float = 1.0 / (1.0 + math.exp(-1.0 / MAGMA_TAU))
CUDA_GRAPH_WARMUP_STEPS: int = 2


_GradientSignature = tuple[int, ...]


@dataclass(slots=True)
class _CudaGraphStep:
    """Captured optimizer step for one gradient-owner signature."""

    graph: torch.cuda.CUDAGraph
    static_grads: tuple[torch.Tensor, ...]


# ============================================================================
# Triton-accelerated symmetric matmul kernel (from flash-muon [4])
# ============================================================================

if TRITON_AVAILABLE:

    def _get_autotune_config():  # noqa: ANN202
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": blk_m,
                    "BLOCK_SIZE_K": blk_k,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=n_stages,
                num_warps=n_warps,
            )
            for blk_m in [32, 64, 128]
            for blk_k in [32, 64]
            for n_stages in [3, 4, 5]
            for n_warps in [4, 8]
        ]

    @triton.autotune(configs=_get_autotune_config(), key=["M", "K"])
    @triton.jit
    def _mmt_kernel(
        x,  # noqa: ANN001
        y,  # noqa: ANN001
        M,  # noqa: ANN001
        K,  # noqa: ANN001
        stride_xm,  # noqa: ANN001
        stride_xk,  # noqa: ANN001
        stride_ym,  # noqa: ANN001
        stride_yn,  # noqa: ANN001
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ) -> None:
        """Compute y = x @ x.T, exploiting symmetry (upper triangle only)."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        # Skip lower triangle — mirror from upper triangle instead
        if pid_m > pid_n:
            return

        offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_xk
            b_ptrs += BLOCK_SIZE_K * stride_xk

        c = accumulator.to(x.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, c, mask=c_mask)

        # Transpose-and-copy: mirror upper triangle to lower
        if pid_m < pid_n:
            ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
            ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
            tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)

    def _matmul_transpose_assign(d_in: torch.Tensor, d_out: torch.Tensor) -> None:
        """Compute d_out = d_in @ d_in.T using triton symmetric matmul kernel."""
        d_in = d_in.contiguous()
        M, K = d_in.shape
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_M"]),
        )
        with torch.cuda.device(d_in.device.index):
            _mmt_kernel[grid](
                d_in,
                d_out,
                M,
                K,
                d_in.stride(0),
                d_in.stride(1),
                d_out.stride(0),
                d_out.stride(1),
            )


# ============================================================================
# Flash Newton-Schulz orthogonalization (triton-accelerated)
# ============================================================================


def _flash_newton_schulz_orth(
    G: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonalize a 2D matrix via two-stage Newton-Schulz with triton-accelerated
    symmetric matmul. Mathematically equivalent to ``_newton_schulz_orth``
    (same DeepSeek-V4 hybrid schedule: ``NS_STEPS_FAST`` fast iters followed
    by ``NS_STEPS_POLISH`` polish iters).

    Parameters
    ----------
    G : torch.Tensor
        Input 2D gradient/update matrix with shape (m, n).
    buf1 : torch.Tensor
        Pre-allocated buffer with shape (M, M) where M = min(m, n), in bfloat16.
    buf2 : torch.Tensor
        Pre-allocated buffer with shape (M, M) where M = min(m, n), in bfloat16.

    Returns
    -------
    torch.Tensor
        Orthogonalized matrix in bfloat16 with shape (m, n).
    """
    # === Step 1. Cast to bf16 and transpose tall matrices ===
    X = G.to(dtype=torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    # === Step 2. Normalize Frobenius norm to at most 1 ===
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=NS_EPS)

    # === Step 3. Fast stage: drive sigma close to 1 ===
    fast_a, fast_b, fast_c = NS_COEFF_FAST
    for _ in range(NS_STEPS_FAST):
        _matmul_transpose_assign(X, buf1)  # buf1 = X @ X.T = A
        _matmul_transpose_assign(buf1, buf2)  # buf2 = A @ A.T = A² (A symmetric)
        B = fast_b * buf1 + fast_c * buf2
        X = fast_a * X + B @ X

    # === Step 4. Polish stage: exact Newton iteration around sigma=1 ===
    polish_a, polish_b, polish_c = NS_COEFF_POLISH
    for _ in range(NS_STEPS_POLISH):
        _matmul_transpose_assign(X, buf1)
        _matmul_transpose_assign(buf1, buf2)
        B = polish_b * buf1 + polish_c * buf2
        X = polish_a * X + B @ X

    # === Step 5. Transpose back if needed ===
    if transposed:
        X = X.transpose(-2, -1)

    return X


def _newton_schulz_orth(
    G: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonalize a 2D matrix via two-stage Newton-Schulz iteration.

    Mathematical formulation:
        X_0 = G / ||G||_F
        X_{k+1} = a*X_k + (b*A_k + c*A_k^2) @ X_k,  where A_k = X_k @ X_k^T

    Two-stage schedule (DeepSeek-V4 §2.4):
        * Fast stage (``NS_STEPS_FAST`` iters, ``NS_COEFF_FAST``):
          a=3.4445, b=-4.7750, c=2.0315 — rapid convergence to sigma ≈ 1.
        * Polish stage (``NS_STEPS_POLISH`` iters, ``NS_COEFF_POLISH``):
          a=2, b=-1.5, c=0.5 — exact Newton iteration pinning sigma to 1.
    """
    # === Step 1. Cast to bf16 and transpose tall matrices ===
    X = G.to(dtype=torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    # === Step 2. Normalize Frobenius norm to at most 1 ===
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=NS_EPS)

    # === Step 3. Fast stage: drive sigma close to 1 ===
    fast_a, fast_b, fast_c = NS_COEFF_FAST
    for _ in range(NS_STEPS_FAST):
        A = torch.mm(X, X.transpose(-2, -1))
        gram_update = torch.addmm(A, A, A, beta=fast_b, alpha=fast_c)
        X = torch.addmm(X, gram_update, X, beta=fast_a, alpha=1.0)

    # === Step 4. Polish stage: exact Newton iteration around sigma=1 ===
    polish_a, polish_b, polish_c = NS_COEFF_POLISH
    for _ in range(NS_STEPS_POLISH):
        A = torch.mm(X, X.transpose(-2, -1))
        gram_update = torch.addmm(A, A, A, beta=polish_b, alpha=polish_c)
        X = torch.addmm(X, gram_update, X, beta=polish_a, alpha=1.0)

    # === Step 5. Transpose back if needed ===
    if transposed:
        X = X.transpose(-2, -1)

    return X


def _batched_newton_schulz_orth(
    G: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonalize a batch of matrices via two-stage Newton-Schulz iteration.

    Uses the same DeepSeek-V4 hybrid schedule as ``_newton_schulz_orth``:
    ``NS_STEPS_FAST`` fast iters with ``NS_COEFF_FAST`` followed by
    ``NS_STEPS_POLISH`` polish iters with ``NS_COEFF_POLISH``.

    Runs as plain eager launches: the whole optimizer step is captured into
    one CUDA graph, which absorbs the launch overhead of every iteration.

    Parameters
    ----------
    G : torch.Tensor
        Input tensor with shape (B, m, n), where B is batch size.

    Returns
    -------
    torch.Tensor
        Orthogonalized tensor in bfloat16 with shape (B, m, n).
    """
    # === Step 1. Validate and prepare matrix orientation ===
    if G.ndim != 3:
        raise ValueError("Batched Newton-Schulz expects a 3D tensor (B, m, n).")

    X = G.to(dtype=torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    # === Step 2. Normalize each slice by Frobenius norm ===
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=NS_EPS)

    # === Step 3. Fast stage: drive sigma close to 1 ===
    fast_a, fast_b, fast_c = NS_COEFF_FAST
    for _ in range(NS_STEPS_FAST):
        A = torch.bmm(X, X.transpose(-2, -1))
        gram_update = torch.baddbmm(A, A, A, beta=fast_b, alpha=fast_c)
        X = torch.baddbmm(X, gram_update, X, beta=fast_a, alpha=1.0)

    # === Step 4. Polish stage: exact Newton iteration around sigma=1 ===
    polish_a, polish_b, polish_c = NS_COEFF_POLISH
    for _ in range(NS_STEPS_POLISH):
        A = torch.bmm(X, X.transpose(-2, -1))
        gram_update = torch.baddbmm(A, A, A, beta=polish_b, alpha=polish_c)
        X = torch.baddbmm(X, gram_update, X, beta=polish_a, alpha=1.0)

    # === Step 5. Restore original orientation ===
    if transposed:
        X = X.transpose(-2, -1)

    return X


class _GramNewtonSchulzOrthogonalizer:
    """
    Orthogonalize rectangular matrices with the fixed Gram Newton-Schulz setup
    used by HybridMuon.
    """

    def __init__(self) -> None:
        # Gram path uses NS_EPS (same numerical role as Standard NS norm clamp).
        # It intentionally does NOT share the smaller ADAM_EPS, because the
        # Polar-Express recipe normalizes before its first iteration and a
        # looser floor is safer under fp32→fp16 cast.
        self.ns_epsilon = float(NS_EPS)
        self.ns_coefficients = tuple(
            (float(a), float(b), float(c)) for a, b, c in POLAR_EXPRESS_COEFFICIENTS
        )
        self._restart_iteration_set = frozenset((2,))

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize a tensor of rectangular matrices.

        Runs as plain eager launches: the whole optimizer step is captured
        into one CUDA graph, which absorbs the launch overhead of every
        iteration and forbids nested graph or compiler regions inside.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor with shape ``(m, n)``, ``(batch, m, n)``, or any tensor
            whose last two dimensions are matrix dimensions.

        Returns
        -------
        torch.Tensor
            Orthogonalized tensor with the same shape and dtype as ``X``.
        """
        return self._orthogonalize_impl(X)

    def _orthogonalize_impl(self, X: torch.Tensor) -> torch.Tensor:
        # === Step 1. Canonicalize leading batch dimensions ===
        original_shape = X.shape
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.reshape(-1, *X.shape[-2:])

        # === Step 2. Normalize in float32 before Gram iteration ===
        original_dtype = X.dtype
        X = X.to(torch.float32)

        # === Step 3. Work on the wide-matrix orientation ===
        should_transpose = X.size(-2) > X.size(-1)
        if should_transpose:
            X = X.mT

        # === Step 4. Run Gram Newton-Schulz in float16 ===
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + self.ns_epsilon)
        X = X.to(torch.float16)
        X = self._gram_newton_schulz(X)

        # === Step 5. Restore original orientation and dtype ===
        if should_transpose:
            X = X.mT

        return X.to(original_dtype).reshape(original_shape)

    def _gram_newton_schulz(self, X: torch.Tensor) -> torch.Tensor:
        # === Step 1. Initialize R_0 = X_0 @ X_0^T and the batch identity ===
        gram_matrix = torch.bmm(X, X.mT)
        batch_size = gram_matrix.size(0)
        identity = (
            torch.eye(
                gram_matrix.size(-1),
                device=X.device,
                dtype=X.dtype,
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        transform = None

        for idx, (coef_a, coef_b, coef_c) in enumerate(self.ns_coefficients):
            # === Step 2. Apply the configured restart boundary ===
            if idx in self._restart_iteration_set and idx != 0:
                X = torch.bmm(transform, X)
                gram_matrix = torch.bmm(X, X.mT)
                transform = None

            # === Step 3. Build Z_k = b * R_k + c * R_k^2 ===
            poly = torch.baddbmm(
                gram_matrix,
                gram_matrix,
                gram_matrix,
                beta=coef_b,
                alpha=coef_c,
            )
            # === Step 4. Accumulate Q_k ===
            if idx == 0 or idx in self._restart_iteration_set:
                # Q_k = Z_k + a * I
                transform = poly + coef_a * identity
            else:
                # Q_k = Q_{k-1} @ (Z_k + a * I) = a * Q_{k-1} + Q_{k-1} @ Z_k
                transform = torch.baddbmm(
                    transform,
                    transform,
                    poly,
                    beta=coef_a,
                    alpha=1.0,
                )

            if (
                idx < len(self.ns_coefficients) - 1
                and idx + 1 not in self._restart_iteration_set
            ):
                # RZ_k = a * R_k + R_k @ Z_k
                gram_poly = torch.baddbmm(
                    gram_matrix,
                    gram_matrix,
                    poly,
                    beta=coef_a,
                    alpha=1.0,
                )
                # R_{k+1} = a * RZ_k + Z_k @ RZ_k
                gram_matrix = torch.baddbmm(
                    gram_poly,
                    poly,
                    gram_poly,
                    beta=coef_a,
                    alpha=1.0,
                )

        # === Step 5. Apply the accumulated Q_last to the current X ===
        return torch.bmm(transform, X)


def _reshape_update_to_matrix_batch(
    update_tensor: torch.Tensor,
    batch_size: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """
    View one update tensor as a batch of matrices.

    Parameters
    ----------
    update_tensor : torch.Tensor
        Update tensor for a single parameter.
    batch_size : int
        Number of matrix slices represented by the parameter.
    rows : int
        Matrix row count for each slice.
    cols : int
        Matrix column count for each slice.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(batch_size, rows, cols)``.
    """
    if update_tensor.is_contiguous():
        return update_tensor.view(batch_size, rows, cols)
    return update_tensor.reshape(batch_size, rows, cols).contiguous()


def _compute_muon_nesterov_updates(
    gradients: list[torch.Tensor],
    momentum_buffers: list[torch.Tensor],
    momentum: float,
    use_foreach: bool = False,
) -> list[torch.Tensor]:
    """
    Update Muon momentum buffers and return Nesterov updates.

    Parameters
    ----------
    gradients : list[torch.Tensor]
        Gradient tensors routed to the Muon path.
    momentum_buffers : list[torch.Tensor]
        Momentum buffers associated with ``gradients``.
    momentum : float
        Momentum coefficient.
    use_foreach : bool
        Use ``torch._foreach_*`` multi-tensor kernels when True.

    Returns
    -------
    list[torch.Tensor]
        Nesterov-style Muon updates with the same shapes as ``gradients``.
    """
    if use_foreach and len(gradients) > 1:
        # m_t = beta * m_{t-1} + (1 - beta) * g_t
        torch._foreach_lerp_(momentum_buffers, gradients, 1.0 - momentum)
        # update = lerp(g_t, m_t, beta) = beta * m_t + (1 - beta) * g_t
        return torch._foreach_lerp(gradients, momentum_buffers, momentum)
    # m_t = beta * m_{t-1} + (1 - beta) * g_t
    for momentum_buffer, grad in zip(momentum_buffers, gradients, strict=True):
        momentum_buffer.lerp_(grad, 1.0 - momentum)
    # update = beta * m_t + (1 - beta) * g_t
    return [
        torch.lerp(grad, momentum_buffer, momentum)
        for grad, momentum_buffer in zip(gradients, momentum_buffers, strict=True)
    ]


def get_adam_route(
    param_name: str | None,
) -> str:
    """
    Determine the optimizer route for a parameter based on its name.

    Parameters
    ----------
    param_name : str | None
        Parameter name. If None, fallback behavior treats parameter as
        matrix (Muon-eligible).

    Returns
    -------
    str
        ``"muon"`` if this parameter is eligible as matrix weight by name,
        ``"adam"`` for Adam path (no weight decay),
        ``"adamw"`` for AdamW path (decoupled weight decay).

    Notes
    -----
    Name-based routing rules (case-insensitive, applied to the final
    effective name segment after stripping trailing numeric ParameterList
    indices):

    1. Is ``"b"`` or contains ``"bias"`` -> ``"adam"`` (no weight decay).
    2. Starts with ``"adam_"`` -> ``"adam"`` (no weight decay).
       Typical: norm scales, radial frequencies.
    3. Starts with ``"adamw_"`` -> ``"adamw"`` (decoupled weight decay).
       Typical: LayerScale parameters.
    4. Otherwise -> ``"muon"`` (eligible for Muon).
    """
    if param_name is None:
        return "muon"
    param_name_lower = param_name.lower()
    name_segments = param_name_lower.split(".")
    leaf_name_idx = len(name_segments) - 1
    while leaf_name_idx > 0 and name_segments[leaf_name_idx].isdigit():
        leaf_name_idx -= 1
    leaf_name = name_segments[leaf_name_idx]
    if leaf_name == "b" or "bias" in leaf_name:
        return "adam"
    if leaf_name.startswith("adam_"):
        return "adam"
    if leaf_name.startswith("adamw_"):
        return "adamw"
    return "muon"


def get_effective_shape(
    shape: torch.Size | tuple[int, ...],
) -> tuple[int, ...]:
    """
    Remove singleton dimensions from a tensor shape for routing decisions.

    Parameters
    ----------
    shape
        Original tensor shape.

    Returns
    -------
    tuple[int, ...]
        Shape without dimensions equal to 1.
        If all dims are 1, returns ``(1,)``.
    """
    effective = tuple(int(dim) for dim in shape if int(dim) != 1)
    if len(effective) == 0:
        return (1,)
    return effective


def get_matrix_view_shape(
    effective_shape: tuple[int, ...],
    muon_mode: str,
) -> tuple[int, int, int] | None:
    """
    Derive Muon matrix-view shape from effective tensor shape.

    Parameters
    ----------
    effective_shape
        Shape with singleton dimensions removed.
    muon_mode
        One of {"2d", "flat", "slice"}.

    Returns
    -------
    tuple[int, int, int] | None
        ``(batch_size, rows, cols)`` when Muon is applicable, otherwise ``None``.
    """
    if len(effective_shape) < 2:
        return None

    if muon_mode == "2d":
        if len(effective_shape) != 2:
            return None
        return (1, int(effective_shape[-2]), int(effective_shape[-1]))
    if muon_mode == "flat":
        rows = int(math.prod(effective_shape[:-1]))
        cols = int(effective_shape[-1])
        return (1, rows, cols)
    if muon_mode == "slice":
        if len(effective_shape) == 2:
            return (1, int(effective_shape[-2]), int(effective_shape[-1]))
        batch_size = int(math.prod(effective_shape[:-2]))
        rows = int(effective_shape[-2])
        cols = int(effective_shape[-1])
        return (batch_size, rows, cols)
    raise ValueError(f"Invalid muon_mode '{muon_mode}'. Use '2d', 'flat', or 'slice'.")


class HybridMuonOptimizer(Optimizer):
    """
    HybridMuon optimizer with 1D Adam path and matrix Muon path.

    This optimizer applies different update rules based on parameter dimensionality,
    parameter names, and ``muon_mode``:
    - Parameters with final effective name segment equal to ``b``, containing
      ``bias``, or starting with ``adam_`` (case-insensitive): standard Adam
      update.
    - Parameters with final effective name segment starting with ``adamw_``
      (case-insensitive): Adam with decoupled weight decay (AdamW-style).
    - 1D parameters: standard Adam update.
    - Parameters are routed by effective shape (singleton dimensions removed).
    - ``muon_mode="2d"``:
      - effective rank 2 parameters use Muon.
      - effective rank >2 parameters use Adam.
    - ``muon_mode="flat"``:
      - effective rank >=2 parameters use flattened matrix-view Muon.
    - ``muon_mode="slice"``:
      - effective rank 2 parameters use Muon.
      - effective rank >=3 parameters apply Muon independently on each trailing
        ``(m, n)`` slice.

    Naming convention for explicit Adam routing:
    - Parameters representing bias terms should use ``b`` or include ``bias``
      in their final effective name segment (case-insensitive).
    - Parameters that are not semantic bias but should still use Adam should
      use an ``adam_`` prefix in their final effective name segment
      (case-insensitive).
    - Parameters that should use Adam with decoupled weight decay should use
      an ``adamw_`` prefix in their final effective name segment
      (case-insensitive).

    This hybrid approach is effective because Muon's orthogonalization is designed
    for weight matrices, while Adam is more suitable for biases and normalization params.

    Update Rules
    ------------
    Muon (>=2D params):
        1. Momentum update: m_t = beta*m_{t-1} + (1-beta)*g_t
        2. Nesterov lookahead: update = beta*m_t + (1-beta)*g_t
        3. Newton-Schulz orthogonalization: orth = NS(update)
        4. Scaling: scale = coeff*sqrt(max(m,n)) or sqrt(max(1, m/n))
        5. Parameter update: theta -= lr * scale * orth

    Adam:
        Standard Adam with bias correction, all computations in float32.
        Decoupled weight decay is applied only to >=2D Adam-routed parameters.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize.
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient for Muon with default 0.95.
    weight_decay : float
        Weight decay coefficient with default 0.001.
        Applied to Muon-routed parameters and >=2D Adam-routed parameters
        with AdamW-style decoupled decay. Not applied to 1D Adam parameters.
    adam_betas : tuple[float, float]
        Adam beta coefficients with default (0.9, 0.95).
    lr_adjust : float
        Learning rate adjustment mode for Muon scaling and Adam learning rate.
        - If lr_adjust <= 0: use match-RMS scaling for Muon,
          scale = lr_adjust_coeff * sqrt(max(m, n)). Adam uses lr directly.
        - If lr_adjust > 0: use rectangular correction for Muon,
          scale = sqrt(max(1.0, m/n)). Adam uses lr/lr_adjust.
        Default is 0.0 (match-RMS scaling).
    lr_adjust_coeff : float
        Coefficient with default 0.18 for match-RMS scaling when
        ``lr_adjust <= 0``:
        ``scale = lr_adjust_coeff * sqrt(max(m, n))``.
        0.18 is the value calibrated by DeepSeek-V4 so that Muon's
        per-element update RMS matches AdamW's typical RMS, enabling
        reuse of AdamW learning rates across both paths. The Moonlight
        reference uses 0.2; both are empirically viable.
    muon_mode : str
        Muon routing mode with default ``"slice"``.
        - ``"2d"``: only 2D parameters are Muon candidates.
        - ``"flat"``: >=2D parameters use flattened matrix-view routing.
        - ``"slice"``: >=3D parameters use per-slice Muon routing on last two dims.
    named_parameters : iterable[tuple[str, torch.Tensor]] | None
        Optional named parameter iterable used for name-based routing.
        Parameters with final effective name segment equal to ``b``, containing
        ``bias``, or starting with ``adam_`` (case-insensitive) are forced to
        Adam (no weight decay). Parameters starting with ``adamw_`` are forced
        to AdamW-style decoupled decay path.
    enable_gram : bool
        Enable the compiled Gram Newton-Schulz path for rectangular Muon
        matrices. Square matrices continue to use the current standard
        Newton-Schulz implementation. Default is True.
    flash_muon : bool
        Enable triton-accelerated Newton-Schulz orthogonalization.
        Requires triton and CUDA. Falls back to PyTorch implementation
        when triton is unavailable or running on CPU. Ignored when
        ``enable_gram=True``.
        Default is True.
    magma_muon : bool
        Enable Magma-lite damping on Muon updates with default True.
        This computes momentum-gradient cosine alignment per Muon block,
        applies EMA smoothing, and rescales Muon updates in [0.1, 1.0].
        Adam/AdamW paths are unchanged. Empirically beneficial for
        MLIP / SeZM training under heavy-tailed gradient noise from
        conservative-force (second-order) autograd.

    Examples
    --------
    >>> optimizer = HybridMuonOptimizer(model.parameters(), lr=5e-4)
    >>> for epoch in range(epochs):
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 5e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.001,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        lr_adjust: float = 0.0,
        lr_adjust_coeff: float = 0.18,
        muon_mode: str = "slice",
        named_parameters: Iterable[tuple[str, torch.Tensor]] | None = None,
        enable_gram: bool = True,
        flash_muon: bool = True,
        magma_muon: bool = True,
        use_foreach: bool | None = None,
    ) -> None:
        # === Step 1. Validate routing mode ===
        muon_mode = str(muon_mode).lower()
        if muon_mode not in {"2d", "flat", "slice"}:
            raise ValueError(
                f"Invalid muon_mode '{muon_mode}'. Use '2d', 'flat', or 'slice'."
            )

        # === Step 2. Register optimizer defaults ===
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "adam_betas": adam_betas,
            "lr_adjust": lr_adjust,
            "lr_adjust_coeff": lr_adjust_coeff,
            "muon_mode": muon_mode,
            "enable_gram": bool(enable_gram),
            "magma_muon": bool(magma_muon),
        }
        super().__init__(params, defaults)

        # === Step 3. Build parameter id -> name mapping ===
        self._param_name_map: dict[int, str] = {}
        if named_parameters is not None:
            self.set_param_names(named_parameters)

        # Static parameter routing: built once on first step() call.
        self._routing_built = False
        self._routing: list[dict[str, Any]] = []

        # === Step 4. Flash-Muon setup ===
        self._use_flash = flash_muon and TRITON_AVAILABLE
        # Lazily allocated NS iteration buffers, keyed by (M, device)
        self._ns_buffers: dict[
            tuple[int, torch.device],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._gram_orthogonalizer: _GramNewtonSchulzOrthogonalizer | None = None

        # === Step 5. Foreach acceleration ===
        # Defaults to True for single-GPU / DDP / ZeRO-1 (plain tensors). Callers
        # that train under FSDP2 (``fully_shard``) should pass
        # ``use_foreach=False`` explicitly because several ``torch._foreach_*``
        # ops lack DTensor sharding propagation on older PyTorch builds.
        self._use_foreach = self._resolve_foreach(use_foreach)

        # === Step 6. Whole-step CUDA graphs ===
        # The optimizer update is host-bound, so each gradient-owner signature
        # is captured after its own eager warmup and replayed thereafter.
        # Signatures share one graph memory pool and one static gradient buffer
        # per parameter: task-specific graphs are mutually exclusive and run on
        # the same stream, so their temporary allocations may safely alias.
        # A fixed Adam owner set uses one clock per parameter group. The first
        # owner-set change materializes equivalent per-parameter clocks, which
        # preserve eager semantics without slowing the common single-task path.
        # Parameters that are not plain CUDA tensors execute the same update
        # eagerly.
        self._graph_enabled = True
        self._graphs: dict[_GradientSignature, _CudaGraphStep] = {}
        self._graph_warmups: dict[_GradientSignature, int] = {}
        self._graph_params: tuple[torch.Tensor, ...] = ()
        self._static_grad_buffers: list[torch.Tensor | None] = []
        self._graph_pool: Any | None = None
        self._graph_capture_stream: torch.cuda.Stream | None = None
        self._adam_ones: dict[torch.device, torch.Tensor] = {}
        self._adam_param_indices: frozenset[int] = frozenset()
        self._adam_signature: _GradientSignature | None = None
        self._per_parameter_adam_clock = False
        self._bias_corrections_migrated = False

    def set_param_names(
        self, named_parameters: Iterable[tuple[str, torch.Tensor]]
    ) -> None:
        """
        Set runtime-only parameter names used for name-based routing.

        The mapping intentionally stays outside optimizer defaults and
        ``param_groups`` so optimizer checkpoints do not persist full
        ``(name, Parameter)`` tuples. Under ZeRO-1 this avoids gathering a
        duplicate model-sized object graph during ``consolidate_state_dict``.
        """
        self._param_name_map = {
            id(param): str(name) for name, param in named_parameters
        }
        self._routing_built = False

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load optimizer state and invalidate captured runtime state.

        Parameters
        ----------
        state_dict : dict[str, Any]
            Optimizer state returned by :meth:`state_dict`.
        """
        super().load_state_dict(state_dict)
        self._adam_signature = None
        self._per_parameter_adam_clock = False
        self._bias_corrections_migrated = False
        self._clear_cuda_graphs()

    def _clear_cuda_graphs(self) -> None:
        """Discard captures whose tensor addresses or routing may be stale."""
        self._graphs.clear()
        self._graph_warmups.clear()
        self._static_grad_buffers = [None] * len(self._graph_params)
        self._graph_pool = None
        self._graph_capture_stream = None

    @staticmethod
    def _resolve_foreach(use_foreach: bool | None) -> bool:
        """Resolve the ``use_foreach`` flag for ``torch._foreach_*`` kernels.

        Foreach fuses per-parameter loops into single kernel launches,
        eliminating Python overhead. When ``use_foreach`` is ``None`` the
        default is ``True`` because plain ``torch.Tensor`` (single-GPU, DDP,
        ZeRO-1) always supports these ops; callers that hit DTensor dispatch
        errors under FSDP2 must pass ``use_foreach=False`` explicitly.
        """
        if use_foreach is not None:
            return bool(use_foreach)
        return True

    def _compute_magma_scales_merged(
        self,
        bucket_entries: list[
            tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        rows: int,
        cols: int,
    ) -> list[torch.Tensor]:
        """Compute Magma-lite scales for a merged bucket with variable batch_sizes.

        Like ``_compute_magma_scales_for_bucket`` but handles entries whose
        ``batch_size`` may differ (produced by the merged-bucket strategy that
        keys on ``(rows, cols)`` instead of ``(batch_size, rows, cols)``).
        """
        n = len(bucket_entries)
        if n == 0:
            return []
        if n == 1:
            entry, _, grad, momentum_buffer = bucket_entries[0]
            return [
                self._compute_magma_scale(
                    param=entry["param"],
                    grad=grad,
                    momentum_buffer=momentum_buffer,
                    batch_size=entry["batch_size"],
                    rows=rows,
                    cols=cols,
                )
            ]

        flat_dim = rows * cols
        grad_views: list[torch.Tensor] = []
        momentum_views: list[torch.Tensor] = []
        entry_batch_sizes: list[int] = []
        for entry, _, grad, momentum_buffer in bucket_entries:
            bs = entry["batch_size"]
            grad_views.append(grad.reshape(bs, flat_dim).to(dtype=torch.float32))
            momentum_views.append(
                momentum_buffer.reshape(bs, flat_dim).to(dtype=torch.float32)
            )
            entry_batch_sizes.append(bs)

        grad_cat = torch.cat(grad_views, dim=0)
        momentum_cat = torch.cat(momentum_views, dim=0)

        dot = (momentum_cat * grad_cat).sum(dim=1)
        denom = (momentum_cat.norm(dim=1) * grad_cat.norm(dim=1)).clamp(min=MAGMA_EPS)
        cosine = (dot / denom).clamp(-1.0, 1.0)
        raw_sigmoid = torch.sigmoid(cosine / MAGMA_TAU)
        raw_scores = (
            (raw_sigmoid - MAGMA_SIGMOID_MIN) / (MAGMA_SIGMOID_MAX - MAGMA_SIGMOID_MIN)
        ).clamp(0.0, 1.0)

        scales: list[torch.Tensor] = []
        offset = 0
        for idx, (entry, _, _, _) in enumerate(bucket_entries):
            bs = entry_batch_sizes[idx]
            param = entry["param"]
            state = self.state[param]
            magma_score = state.get("magma_score")
            if (
                magma_score is None
                or magma_score.ndim != 1
                or magma_score.numel() != bs
                or magma_score.device != param.device
            ):
                magma_score = torch.full(
                    (bs,), 0.5, dtype=torch.float32, device=param.device
                )
                state["magma_score"] = magma_score
            elif magma_score.dtype != torch.float32:
                magma_score = magma_score.to(dtype=torch.float32, device=param.device)
                state["magma_score"] = magma_score
            magma_score.mul_(MAGMA_EMA_DECAY).add_(
                raw_scores[offset : offset + bs], alpha=(1.0 - MAGMA_EMA_DECAY)
            )
            scales.append(MAGMA_MIN_SCALE + (1.0 - MAGMA_MIN_SCALE) * magma_score)
            offset += bs

        return scales

    def _compute_magma_scale(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        batch_size: int,
        rows: int,
        cols: int,
    ) -> torch.Tensor:
        """
        Compute Magma-lite Muon damping scales from momentum-gradient alignment.

        Implements a stabilized version of Magma (Momentum-Aligned Gradient Masking)
        adapted for MLIP force-field training. Computes block-wise alignment scores
        between Muon momentum and current gradients, applies EMA smoothing, and
        rescales Muon updates to improve stability under heavy-tailed gradient noise.

        Notes
        -----
        For each Muon block b:

        1. Compute cosine similarity between momentum and gradient:

           cos(b) = <μ_t^(b), g_t^(b)> / (||μ_t^(b)|| * ||g_t^(b)||)

        2. Apply sigmoid with range stretching to [0, 1]:

           s_raw^(b) = (sigmoid(cos(b) / τ) - s_min) / (s_max - s_min)

           where τ=2.0, s_min=sigmoid(-1/τ), s_max=sigmoid(1/τ).
           This stretches the narrow sigmoid range [0.38, 0.62] to [0, 1].

        3. Apply EMA smoothing:

           s̃_t^(b) = a * s̃_{t-1}^(b) + (1-a) * s_raw^(b)

           where a=0.9 (MAGMA_EMA_DECAY).

        4. Map to damping scale in [s_min_scale, 1.0]:

           scale^(b) = s_min_scale + (1 - s_min_scale) * s̃_t^(b)

           where s_min_scale=0.1 (MAGMA_MIN_SCALE).

        5. Apply damping to Muon update:

           Δ̃^(b) = scale^(b) * Δ^(b)  (soft scaling, no Bernoulli masking)

        Key differences from the original Magma paper:

        - Sigmoid range stretching: Paper uses raw sigmoid with narrow range [0.38, 0.62].
          We stretch to [0, 1] for better discrimination between aligned/misaligned blocks.
        - Soft scaling: Paper uses Bernoulli masking (50% skip probability).
          We use continuous soft scaling [0.1, 1.0] for stability in MLIP training.
        - Minimum scale: Paper allows scale=0 (complete skip).
          We enforce scale >= 0.1 to guarantee minimum learning rate.

        Parameters
        ----------
        param : torch.Tensor
            Parameter updated by Muon.
        grad : torch.Tensor
            Current gradient tensor with shape compatible with ``(batch_size, rows, cols)``.
        momentum_buffer : torch.Tensor
            Muon momentum buffer (updated m_t) with same shape as ``grad``.
        batch_size : int
            Number of Muon blocks (1 for 2d/flat mode, >1 for slice mode).
        rows : int
            Matrix row count per block.
        cols : int
            Matrix column count per block.

        Returns
        -------
        torch.Tensor
            Damping scales with shape (batch_size,) in [MAGMA_MIN_SCALE, 1.0].
        """
        # === Step 1. Restore or initialize EMA score state ===
        # The EMA advances in place on the persistent state tensor: a fresh
        # tensor rebound into the state dict is a host-side assignment that a
        # captured CUDA graph executes only at capture time, which would
        # freeze the EMA recursion at that step's value on every replay.
        state = self.state[param]
        magma_score = state.get("magma_score")
        if (
            magma_score is None
            or magma_score.ndim != 1
            or magma_score.numel() != batch_size
            or magma_score.device != param.device
        ):
            magma_score = torch.full(
                (batch_size,),
                0.5,
                dtype=torch.float32,
                device=param.device,
            )
            state["magma_score"] = magma_score
        elif magma_score.dtype != torch.float32:
            magma_score = magma_score.to(dtype=torch.float32, device=param.device)
            state["magma_score"] = magma_score

        # === Step 2. Build matrix-view for block-wise cosine ===
        grad_view = grad.reshape(batch_size, rows, cols).reshape(batch_size, -1)
        momentum_view = momentum_buffer.reshape(batch_size, rows, cols).reshape(
            batch_size, -1
        )
        grad_view = grad_view.to(dtype=torch.float32)
        momentum_view = momentum_view.to(dtype=torch.float32)

        # === Step 3. Compute cosine alignment with numerical protection ===
        dot = (momentum_view * grad_view).sum(dim=1)
        denom = (momentum_view.norm(dim=1) * grad_view.norm(dim=1)).clamp(min=MAGMA_EPS)
        cosine = (dot / denom).clamp(min=-1.0, max=1.0)

        # === Step 4. Sigmoid mapping + range stretching to [0, 1] ===
        raw_sigmoid = torch.sigmoid(cosine / MAGMA_TAU)
        raw_score = (raw_sigmoid - MAGMA_SIGMOID_MIN) / (
            MAGMA_SIGMOID_MAX - MAGMA_SIGMOID_MIN
        )
        raw_score = raw_score.clamp(min=0.0, max=1.0)

        # === Step 5. Update EMA score and convert to damping scale ===
        magma_score.mul_(MAGMA_EMA_DECAY).add_(raw_score, alpha=1.0 - MAGMA_EMA_DECAY)
        return MAGMA_MIN_SCALE + (1.0 - MAGMA_MIN_SCALE) * magma_score

    def _compute_magma_scales_for_bucket(
        self,
        bucket_entries: list[
            tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        batch_size: int,
        rows: int,
        cols: int,
    ) -> list[torch.Tensor]:
        """
        Compute Magma-lite damping scales for one Muon bucket in a batched way.

        Parameters
        ----------
        bucket_entries : list[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]]
            Bucket entries as ``(entry, update_tensor, grad, momentum_buffer)``.
        batch_size : int
            Number of Muon blocks per parameter in this bucket.
        rows : int
            Matrix row count for this bucket.
        cols : int
            Matrix column count for this bucket.

        Returns
        -------
        list[torch.Tensor]
            Magma scales for each bucket entry. Each tensor has shape (batch_size,).
        """
        # === Step 0. Fast path for single-entry bucket ===
        if len(bucket_entries) == 1:
            entry, _update_tensor, grad, momentum_buffer = bucket_entries[0]
            return [
                self._compute_magma_scale(
                    param=entry["param"],
                    grad=grad,
                    momentum_buffer=momentum_buffer,
                    batch_size=batch_size,
                    rows=rows,
                    cols=cols,
                )
            ]

        # === Step 1. Build batched matrix views ===
        grad_views: list[torch.Tensor] = []
        momentum_views: list[torch.Tensor] = []
        for _, _, grad, momentum_buffer in bucket_entries:
            grad_view = grad.reshape(batch_size, rows, cols).reshape(batch_size, -1)
            momentum_view = momentum_buffer.reshape(batch_size, rows, cols).reshape(
                batch_size, -1
            )
            grad_views.append(grad_view.to(dtype=torch.float32))
            momentum_views.append(momentum_view.to(dtype=torch.float32))

        grad_batch = torch.stack(grad_views, dim=0)
        momentum_batch = torch.stack(momentum_views, dim=0)

        # === Step 2. Compute cosine alignment for all entries ===
        dot = (momentum_batch * grad_batch).sum(dim=2)
        denom = (momentum_batch.norm(dim=2) * grad_batch.norm(dim=2)).clamp(
            min=MAGMA_EPS
        )
        cosine = (dot / denom).clamp(min=-1.0, max=1.0)
        raw_sigmoid = torch.sigmoid(cosine / MAGMA_TAU)
        raw_scores = (raw_sigmoid - MAGMA_SIGMOID_MIN) / (
            MAGMA_SIGMOID_MAX - MAGMA_SIGMOID_MIN
        )
        raw_scores = raw_scores.clamp(min=0.0, max=1.0)

        # === Step 3. Update per-parameter EMA score state ===
        scales: list[torch.Tensor] = []
        for idx, (entry, _, _, _) in enumerate(bucket_entries):
            param = entry["param"]
            state = self.state[param]
            magma_score = state.get("magma_score")
            if (
                magma_score is None
                or magma_score.ndim != 1
                or magma_score.numel() != batch_size
                or magma_score.device != param.device
            ):
                magma_score = torch.full(
                    (batch_size,),
                    0.5,
                    dtype=torch.float32,
                    device=param.device,
                )
                state["magma_score"] = magma_score
            elif magma_score.dtype != torch.float32:
                magma_score = magma_score.to(dtype=torch.float32, device=param.device)
                state["magma_score"] = magma_score

            magma_score.mul_(MAGMA_EMA_DECAY).add_(
                raw_scores[idx], alpha=(1.0 - MAGMA_EMA_DECAY)
            )
            scales.append(MAGMA_MIN_SCALE + (1.0 - MAGMA_MIN_SCALE) * magma_score)

        return scales

    def _get_ns_buffers(
        self,
        M: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get or lazily allocate pre-allocated buffers for flash Newton-Schulz.

        Parameters
        ----------
        M : int
            Square buffer dimension (= min(rows, cols) of the update matrix).
        device : torch.device
            Target CUDA device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (buf1, buf2), each with shape (M, M) in bfloat16.
        """
        key = (M, device)
        if key not in self._ns_buffers:
            self._ns_buffers[key] = (
                torch.empty(M, M, dtype=torch.bfloat16, device=device),
                torch.empty(M, M, dtype=torch.bfloat16, device=device),
            )
        return self._ns_buffers[key]

    def _get_gram_orthogonalizer(self) -> _GramNewtonSchulzOrthogonalizer:
        """
        Lazily initialize the compiled Gram orthogonalizer.

        Returns
        -------
        _GramNewtonSchulzOrthogonalizer
            Shared Gram orthogonalizer instance for the optimizer.
        """
        if self._gram_orthogonalizer is None:
            self._gram_orthogonalizer = _GramNewtonSchulzOrthogonalizer()
        return self._gram_orthogonalizer

    def _process_merged_gram_buckets(
        self,
        gram_buckets: dict[
            tuple[int, int, torch.device, torch.dtype],
            list[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]],
        ],
        lr_adjust: float,
        lr_adjust_coeff: float,
        magma_scales_map: dict[int, torch.Tensor],
        out_params: list[torch.Tensor],
        out_deltas: list[torch.Tensor],
    ) -> None:
        """Column-pad merge across rectangular buckets sharing the same min_dim.

        Rectangular Muon matrices with the same ``min(rows, cols)`` can be
        fused into a single Gram Newton-Schulz call by zero-padding the
        **column** (large) dimension to the group maximum.  This reduces the
        number of compiled Gram NS dispatches and improves GPU occupancy.

        Mathematical equivalence proof for column-padding:
        Both Standard NS and Gram NS operate on the wide orientation
        ``X  (m x n)``, ``m <= n``.  The Gram matrix is
        ``R = X @ X^T  (m x m)``.

        Let ``X_pad = [X | 0]  (m x (n+p))`` where the last p columns are
        zero.  Then:

        1. Frobenius norm is unchanged:
           ``||X_pad||_F = ||X||_F``
           because the zero columns contribute nothing.

        2. Gram matrix is unchanged:
           ``R_pad = X_pad @ X_pad^T = X @ X^T + 0 @ 0^T = R``

        3. Since all NS iterations (both standard quintic and Gram/Polar-
           Express) depend *only* on R (which is m x m regardless of n),
           every intermediate ``Q_k`` is identical.

        4. The restart step ``X_new = Q @ X_pad = [Q @ X | 0]`` also
           preserves the invariant ``R_new = Q @ R @ Q^T``, so subsequent
           iterations remain identical.

        5. The final output is ``Q_last @ X_pad = [Q_last @ X | 0]``.
           Truncating to the first n columns exactly recovers the
           unpadded result.

        **Constraint**: Only the *column* (large) dimension may be padded.
        Padding rows would change the size of R and break equivalence.

        Per-entry ``scale`` and Magma damping are applied *after* unpadding,
        since different original shapes have different ``max(rows, cols)``.
        The finished deltas are appended to ``out_params`` / ``out_deltas``;
        the caller applies every route's update as one multi-tensor kernel.
        """
        # --- Group rectangular buckets by (min_dim, device, dtype) ---
        super_buckets: dict[
            tuple[int, torch.device, torch.dtype],
            list[
                tuple[
                    int,
                    int,
                    bool,
                    list[
                        tuple[
                            dict[str, Any],
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                        ]
                    ],
                ]
            ],
        ] = {}

        for (rows, cols, dev, dt), bucket_entries in gram_buckets.items():
            min_dim = min(rows, cols)
            transposed = rows > cols
            sb_key = (min_dim, dev, dt)
            if sb_key not in super_buckets:
                super_buckets[sb_key] = []
            super_buckets[sb_key].append((rows, cols, transposed, bucket_entries))

        gram_orth = self._get_gram_orthogonalizer()

        for (_min_dim, _dev, _dt), sub_list in super_buckets.items():
            # Find the maximum large-dimension across all sub-buckets.
            padded_max_dim = max(max(r, c) for r, c, _, _ in sub_list)

            # Collect all matrices in wide orientation (min_dim, padded_max_dim).
            all_wide: list[torch.Tensor] = []
            # Track per-matrix metadata for the split-back phase.
            # Each entry: (param_entry, batch_size, orig_max_dim, was_transposed)
            all_meta: list[tuple[dict[str, Any], int, int, bool]] = []

            for rows, cols, was_transposed, bucket_entries in sub_list:
                orig_max_dim = max(rows, cols)
                for entry, update_tensor, _, _ in bucket_entries:
                    bs = entry["batch_size"]
                    mat = _reshape_update_to_matrix_batch(update_tensor, bs, rows, cols)
                    # Orient to wide: (bs, min_dim, orig_max_dim)
                    if was_transposed:
                        mat = mat.transpose(-2, -1)
                    # Pad columns if needed: (bs, min_dim, padded_max_dim)
                    pad_width = padded_max_dim - orig_max_dim
                    if pad_width > 0:
                        mat = torch.nn.functional.pad(mat, (0, pad_width))
                    all_wide.append(mat)
                    all_meta.append((entry, bs, orig_max_dim, was_transposed))

            # Single Gram NS call on the entire super-bucket.
            # Shape: (total_batch, min_dim, padded_max_dim)
            stacked = torch.cat(all_wide, dim=0)
            orthogonalized = gram_orth(stacked)

            # Split back, unpad, un-transpose, apply scale + Magma + update.
            offset = 0
            for entry, bs, orig_max_dim, was_transposed in all_meta:
                orth_slice = orthogonalized[offset : offset + bs]
                offset += bs

                # Unpad: keep only the first orig_max_dim columns.
                if orig_max_dim < padded_max_dim:
                    orth_slice = orth_slice[:, :, :orig_max_dim]

                # Un-transpose back to original (rows, cols) orientation.
                if was_transposed:
                    orth_slice = orth_slice.transpose(-2, -1)

                # Per-entry scale (depends on original max(rows, cols)).
                orig_rows, orig_cols = entry["rows"], entry["cols"]
                if lr_adjust <= 0:
                    scale = lr_adjust_coeff * math.sqrt(
                        float(max(orig_rows, orig_cols))
                    )
                else:
                    scale = max(1.0, orig_rows / orig_cols) ** 0.5
                orth_slice = orth_slice * scale

                # Per-entry Magma damping.
                magma_scale = magma_scales_map.get(id(entry["param"]))
                if magma_scale is not None:
                    orth_slice = orth_slice * magma_scale.view(bs, 1, 1).to(
                        dtype=orth_slice.dtype, device=orth_slice.device
                    )

                out_params.append(entry["param"])
                out_deltas.append(orth_slice.reshape(entry["param"].shape))

    def _build_param_routing(self) -> None:
        """
        Classify parameters into Muon, Adam, and AdamW routes (static routing).

        Routing logic:
        - name-based ``b``, ``bias``, or ``adam_`` route → Adam (no decay)
        - name-based ``adamw_`` prefix → AdamW (decoupled weight decay)
        - effective shape rank <2 → Adam (no decay)
        - non-matrix effective shape for current muon_mode → AdamW (decoupled)
        - remaining eligible matrix params → Muon path
        """
        if self._routing_built:
            return

        self._clear_cuda_graphs()
        self._routing = []
        for group in self.param_groups:
            muon_params: list[dict[str, Any]] = []
            adam_no_decay: list[dict[str, Any]] = []
            adam_decay: list[dict[str, Any]] = []

            muon_mode = group["muon_mode"]

            for p in group["params"]:
                param_name = self._param_name_map.get(id(p))

                # === Step 1. Name-based explicit route ===
                route = get_adam_route(param_name)
                if route == "adam":
                    adam_no_decay.append({"param": p, "name": param_name})
                    continue
                if route == "adamw":
                    adam_decay.append({"param": p, "name": param_name})
                    continue

                # === Step 2. Effective <2D parameters → Adam ===
                effective_shape = get_effective_shape(p.shape)
                if len(effective_shape) < 2:
                    adam_no_decay.append({"param": p, "name": param_name})
                    continue

                # === Step 3. Non-matrix effective shape in current mode → AdamW-style ===
                matrix_shape = get_matrix_view_shape(effective_shape, muon_mode)
                if matrix_shape is None:
                    adam_decay.append({"param": p, "name": param_name})
                    continue

                # === Step 4. Eligible matrix params → Muon ===
                batch_size, rows, cols = matrix_shape
                muon_params.append(
                    {
                        "param": p,
                        "name": param_name,
                        "batch_size": batch_size,
                        "rows": rows,
                        "cols": cols,
                    }
                )

            self._routing.append(
                {
                    "muon_params": muon_params,
                    "adam_no_decay": adam_no_decay,
                    "adam_decay": adam_decay,
                }
            )

        self._graph_params = tuple(
            p for group in self.param_groups for p in group["params"]
        )
        adam_param_ids = {
            id(entry["param"])
            for route in self._routing
            for key in ("adam_no_decay", "adam_decay")
            for entry in route[key]
        }
        self._adam_param_indices = frozenset(
            index
            for index, param in enumerate(self._graph_params)
            if id(param) in adam_param_ids
        )
        self._static_grad_buffers = [None] * len(self._graph_params)
        self._routing_built = True

    # ------------------------------------------------------------------
    # Foreach-aware helpers for Adam moment updates
    # ------------------------------------------------------------------

    def _adam_update_moments(
        self,
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        grads_fp32: list[torch.Tensor],
        beta1: float,
        beta2: float,
    ) -> None:
        """Update Adam first/second moment estimates, foreach-accelerated when safe.

        exp_avg  = beta1 * exp_avg  + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        """
        if self._use_foreach and len(exp_avgs) > 1:
            torch._foreach_lerp_(exp_avgs, grads_fp32, 1.0 - beta1)
            grad_sq = torch._foreach_mul(grads_fp32, grads_fp32)
            torch._foreach_lerp_(exp_avg_sqs, grad_sq, 1.0 - beta2)
        else:
            for ea, g in zip(exp_avgs, grads_fp32, strict=True):
                ea.lerp_(g, 1.0 - beta1)
            grad_sq = [g * g for g in grads_fp32]
            for eas, gsq in zip(exp_avg_sqs, grad_sq, strict=True):
                eas.lerp_(gsq, 1.0 - beta2)

    def _weight_decay_inplace(
        self,
        params: list[torch.Tensor],
        factor: torch.Tensor,
    ) -> None:
        """Apply multiplicative weight decay, foreach-accelerated when safe.

        ``factor`` is a 0-dim device tensor (``1 - lr * weight_decay``), so
        the decay follows the learning-rate schedule inside a captured graph.
        """
        if self._use_foreach and len(params) > 1:
            torch._foreach_mul_(params, factor)
        else:
            for p in params:
                p.mul_(factor)

    def _ensure_group_tensors(
        self, group: dict[str, Any], device: torch.device
    ) -> None:
        """Materialize device scalars used by eager and captured updates."""
        if "lr_device" not in group:
            group["lr_device"] = torch.zeros((), dtype=torch.float32, device=device)
        elif (
            group["lr_device"].device != device
            or group["lr_device"].dtype != torch.float32
        ):
            group["lr_device"] = group["lr_device"].to(
                device=device, dtype=torch.float32
            )

        if self._per_parameter_adam_clock:
            return
        for key in ("beta1_pow_device", "beta2_pow_device"):
            if key not in group:
                group[key] = torch.ones((), dtype=torch.float32, device=device)
            elif group[key].device != device or group[key].dtype != torch.float32:
                group[key] = group[key].to(device=device, dtype=torch.float32)

    def _adam_one(self, device: torch.device) -> torch.Tensor:
        """Return the cached scalar target for Adam correction recurrences."""
        one = self._adam_ones.get(device)
        if one is None:
            one = torch.ones((), dtype=torch.float32, device=device)
            self._adam_ones[device] = one
        return one

    def _update_adam_bias_corrections(
        self,
        bias_correction1: list[torch.Tensor],
        bias_correction2: list[torch.Tensor],
        beta1: float,
        beta2: float,
    ) -> None:
        """Advance per-parameter Adam corrections with fused EMA kernels."""
        one = self._adam_one(bias_correction1[0].device)
        targets = [one] * len(bias_correction1)
        if self._use_foreach and len(bias_correction1) > 1:
            torch._foreach_lerp_(bias_correction1, targets, 1.0 - beta1)
            torch._foreach_lerp_(bias_correction2, targets, 1.0 - beta2)
            return
        for correction1, correction2 in zip(
            bias_correction1, bias_correction2, strict=True
        ):
            correction1.lerp_(one, 1.0 - beta1)
            correction2.lerp_(one, 1.0 - beta2)

    def _adam_apply_updates(
        self,
        params: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        bias_correction1: list[torch.Tensor],
        bias_correction2: list[torch.Tensor],
        group: dict[str, Any],
        lr_factor: float,
    ) -> None:
        """Apply the bias-corrected Adam update as multi-tensor kernels.

        A fixed owner set broadcasts one correction per parameter group. Once
        the owner set changes, each active parameter supplies its own device
        correction so inactive task heads do not advance their Adam clock.
        """
        if self._use_foreach and len(params) > 1:
            if self._per_parameter_adam_clock:
                step_sizes = torch._foreach_reciprocal(bias_correction1)
                torch._foreach_mul_(step_sizes, group["lr_device"] * lr_factor)
                denom = torch._foreach_div(exp_avg_sqs, bias_correction2)
            else:
                correction1 = 1.0 - group["beta1_pow_device"]
                correction2 = 1.0 - group["beta2_pow_device"]
                step_size = group["lr_device"] * (lr_factor / correction1)
                denom = torch._foreach_div(exp_avg_sqs, correction2)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, ADAM_EPS)
            deltas = torch._foreach_div(exp_avgs, denom)
            if self._per_parameter_adam_clock:
                torch._foreach_mul_(deltas, step_sizes)
            else:
                torch._foreach_mul_(deltas, step_size)

            groups: dict[torch.dtype, list[int]] = {}
            for i, p in enumerate(params):
                groups.setdefault(p.dtype, []).append(i)
            for dtype, idxs in groups.items():
                dtype_deltas = [deltas[i] for i in idxs]
                if dtype is not torch.float32:
                    dtype_deltas = [d.to(dtype) for d in dtype_deltas]
                torch._foreach_add_([params[i] for i in idxs], dtype_deltas, alpha=-1)
        else:
            if self._per_parameter_adam_clock:
                for p, exp_avg, exp_avg_sq, correction1, correction2 in zip(
                    params,
                    exp_avgs,
                    exp_avg_sqs,
                    bias_correction1,
                    bias_correction2,
                    strict=True,
                ):
                    step_size = group["lr_device"] * (lr_factor / correction1)
                    denom = (exp_avg_sq / correction2).sqrt().add_(ADAM_EPS)
                    delta = (exp_avg / denom).mul_(step_size)
                    p.add_(delta.to(p.dtype), alpha=-1)
            else:
                correction1 = 1.0 - group["beta1_pow_device"]
                correction2 = 1.0 - group["beta2_pow_device"]
                for p, exp_avg, exp_avg_sq in zip(
                    params, exp_avgs, exp_avg_sqs, strict=True
                ):
                    step_size = group["lr_device"] * (lr_factor / correction1)
                    denom = (exp_avg_sq / correction2).sqrt().add_(ADAM_EPS)
                    delta = (exp_avg / denom).mul_(step_size)
                    p.add_(delta.to(p.dtype), alpha=-1)

    def _apply_param_deltas(
        self,
        params: list[torch.Tensor],
        deltas: list[torch.Tensor],
        lr_device: torch.Tensor,
    ) -> None:
        """Apply ``p -= lr * delta`` over a route as multi-tensor kernels."""
        if not params:
            return
        if self._use_foreach and len(params) > 1:
            groups: dict[torch.dtype, list[int]] = {}
            for i, p in enumerate(params):
                groups.setdefault(p.dtype, []).append(i)
            for dtype, idxs in groups.items():
                scaled = torch._foreach_mul([deltas[i] for i in idxs], lr_device)
                if dtype is not torch.float32:
                    scaled = [d.to(dtype) for d in scaled]
                torch._foreach_add_([params[i] for i in idxs], scaled, alpha=-1)
        else:
            for p, delta in zip(params, deltas, strict=True):
                p.add_((delta * lr_device).to(p.dtype), alpha=-1)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """
        Perform a single optimization step.

        On CUDA each gradient-owner signature is captured after two eager
        warmup steps and replayed thereafter. This preserves whole-step graph
        acceleration when multi-task training alternates parameter subsets.
        The signatures share static gradient buffers and one graph memory
        pool. Parameters that are not plain CUDA tensors run the identical
        update eagerly.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        torch.Tensor | None
            The loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Build static parameter routing on first call.
        self._build_param_routing()
        self._migrate_bias_corrections()

        signature, grads = self._collect_gradients()
        if not signature:
            return loss
        self._prepare_adam_clock(signature)

        # Host-driven scalars refresh outside every capture or replay.
        device = self.param_groups[0]["params"][0].device
        for group in self.param_groups:
            group_device = group["params"][0].device
            self._ensure_group_tensors(group, group_device)
            group["lr_device"].fill_(float(group["lr"]))

        if not self._graph_supported(device):
            self._step_impl(None)
            return loss

        graph_step = self._graphs.get(signature)
        if graph_step is not None:
            torch._foreach_copy_(graph_step.static_grads, grads)
            graph_step.graph.replay()
            return loss

        warmups = self._graph_warmups.get(signature, 0)
        if warmups < CUDA_GRAPH_WARMUP_STEPS:
            self._graph_warmups[signature] = warmups + 1
            self._step_impl(None)
            return loss

        graph_step = self._capture_cuda_graph(signature, grads, device)
        self._graphs[signature] = graph_step
        self._graph_warmups.pop(signature)
        graph_step.graph.replay()
        return loss

    def _graph_supported(self, device: torch.device) -> bool:
        """Whether the whole-step CUDA graph serves this configuration."""
        if not self._graph_enabled or device.type != "cuda":
            return False
        for group in self.param_groups:
            for p in group["params"]:
                if (
                    type(p) not in (torch.Tensor, torch.nn.Parameter)
                    or p.device != device
                ):
                    return False
        return True

    def _migrate_bias_corrections(self) -> None:
        """Restore either uniform or per-parameter Adam clock state.

        Per-parameter powers from eager checkpoints and corrections from
        dynamic-signature checkpoints select the dynamic clock. A checkpoint
        containing only group powers retains the uniform fast path.
        """
        if self._bias_corrections_migrated:
            return

        has_per_parameter_clock = False
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "exp_avg" not in state:
                    continue
                if any(
                    key in state
                    for key in (
                        "bias_correction1",
                        "bias_correction2",
                        "beta1_pow",
                        "beta2_pow",
                    )
                ):
                    has_per_parameter_clock = True

        if has_per_parameter_clock:
            self._per_parameter_adam_clock = True
            for group in self.param_groups:
                group_beta1_pow = group.pop("beta1_pow_device", None)
                group_beta2_pow = group.pop("beta2_pow_device", None)
                if (group_beta1_pow is None) != (group_beta2_pow is None):
                    raise RuntimeError(
                        "HybridMuon checkpoint contains an incomplete group Adam clock"
                    )
                for p in group["params"]:
                    state = self.state.get(p)
                    if not state or "exp_avg" not in state:
                        continue
                    correction1 = state.get("bias_correction1")
                    correction2 = state.get("bias_correction2")
                    if (correction1 is None) != (correction2 is None):
                        raise RuntimeError(
                            "HybridMuon checkpoint contains an incomplete "
                            "per-parameter Adam correction"
                        )
                    beta1_pow = state.pop("beta1_pow", None)
                    beta2_pow = state.pop("beta2_pow", None)
                    if (beta1_pow is None) != (beta2_pow is None):
                        raise RuntimeError(
                            "HybridMuon checkpoint contains an incomplete "
                            "per-parameter Adam power"
                        )
                    if correction1 is None:
                        beta1_pow = group_beta1_pow if beta1_pow is None else beta1_pow
                        beta2_pow = group_beta2_pow if beta2_pow is None else beta2_pow
                        if beta1_pow is None or beta2_pow is None:
                            raise RuntimeError(
                                "HybridMuon Adam state is missing its clock"
                            )
                        correction1 = 1.0 - torch.as_tensor(
                            beta1_pow, dtype=torch.float32, device=p.device
                        )
                        correction2 = 1.0 - torch.as_tensor(
                            beta2_pow, dtype=torch.float32, device=p.device
                        )
                    state["bias_correction1"] = torch.as_tensor(
                        correction1, dtype=torch.float32, device=p.device
                    ).reshape(())
                    state["bias_correction2"] = torch.as_tensor(
                        correction2, dtype=torch.float32, device=p.device
                    ).reshape(())
            self._adam_signature = None
        else:
            self._per_parameter_adam_clock = False
            for group in self.param_groups:
                beta1_pow = group.get("beta1_pow_device")
                beta2_pow = group.get("beta2_pow_device")
                if (beta1_pow is None) != (beta2_pow is None):
                    raise RuntimeError(
                        "HybridMuon checkpoint contains an incomplete group Adam clock"
                    )
                has_adam_state = any(
                    "exp_avg" in self.state.get(p, {}) for p in group["params"]
                )
                if has_adam_state and beta1_pow is None:
                    raise RuntimeError("HybridMuon Adam state is missing its clock")
                if beta1_pow is not None:
                    device = group["params"][0].device
                    group["beta1_pow_device"] = torch.as_tensor(
                        beta1_pow, dtype=torch.float32, device=device
                    ).reshape(())
                    group["beta2_pow_device"] = torch.as_tensor(
                        beta2_pow, dtype=torch.float32, device=device
                    ).reshape(())
            self._adam_signature = (
                tuple(
                    index
                    for index, param in enumerate(self._graph_params)
                    if index in self._adam_param_indices
                    and "exp_avg" in self.state.get(param, {})
                )
                or None
            )

        self._bias_corrections_migrated = True

    def _prepare_adam_clock(self, signature: _GradientSignature) -> None:
        """Select the exact Adam clock representation for this owner set."""
        if self._per_parameter_adam_clock:
            return
        adam_signature = tuple(
            index for index in signature if index in self._adam_param_indices
        )
        if self._adam_signature is None:
            self._adam_signature = adam_signature
            return
        if adam_signature == self._adam_signature:
            return

        self._materialize_per_parameter_adam_clock()
        self._clear_cuda_graphs()

    def _materialize_per_parameter_adam_clock(self) -> None:
        """Split uniform group clocks without changing any Adam step count."""
        for group in self.param_groups:
            beta1_pow = group.pop("beta1_pow_device", None)
            beta2_pow = group.pop("beta2_pow_device", None)
            if beta1_pow is None or beta2_pow is None:
                raise RuntimeError("HybridMuon group Adam clock is not initialized")
            correction1 = 1.0 - beta1_pow
            correction2 = 1.0 - beta2_pow
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "exp_avg" not in state:
                    continue
                state["bias_correction1"] = torch.as_tensor(
                    correction1.detach().clone(),
                    dtype=torch.float32,
                    device=p.device,
                ).reshape(())
                state["bias_correction2"] = torch.as_tensor(
                    correction2.detach().clone(),
                    dtype=torch.float32,
                    device=p.device,
                ).reshape(())
        self._per_parameter_adam_clock = True
        self._adam_signature = None

    def _collect_gradients(
        self,
    ) -> tuple[_GradientSignature, tuple[torch.Tensor, ...]]:
        """Collect live gradients and their stable parameter indices."""
        signature: list[int] = []
        grads: list[torch.Tensor] = []
        for index, param in enumerate(self._graph_params):
            if param.grad is None:
                continue
            signature.append(index)
            grads.append(param.grad)
        return tuple(signature), tuple(grads)

    def _capture_cuda_graph(
        self,
        signature: _GradientSignature,
        grads: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> _CudaGraphStep:
        """Capture one warmed gradient-owner signature."""
        static_grads: list[torch.Tensor] = []
        grad_map: dict[int, torch.Tensor] = {}
        for index, grad in zip(signature, grads, strict=True):
            static_grad = self._static_grad_buffers[index]
            if static_grad is None:
                static_grad = torch.zeros_like(grad)
                self._static_grad_buffers[index] = static_grad
            static_grads.append(static_grad)
            grad_map[id(self._graph_params[index])] = static_grad

        static_grads_tuple = tuple(static_grads)
        torch._foreach_copy_(static_grads_tuple, grads)

        with torch.cuda.device(device):
            if self._graph_pool is None:
                self._graph_pool = torch.cuda.graph_pool_handle()
                self._graph_capture_stream = torch.cuda.Stream(device=device)
            if self._graph_capture_stream is None:
                raise RuntimeError(
                    "HybridMuon CUDA graph capture stream is not initialized"
                )
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=self._graph_pool,
                stream=self._graph_capture_stream,
            ):
                self._step_impl(grad_map)

        return _CudaGraphStep(graph=graph, static_grads=static_grads_tuple)

    def _step_impl(self, grad_map: dict[int, torch.Tensor] | None) -> None:
        """Run one optimization update over every parameter group.

        ``grad_map`` routes parameter ids to the static gradient buffers when
        the update runs under graph capture; ``None`` reads the live
        ``p.grad`` directly (warmup and the eager fallback).
        """
        for group_idx, group in enumerate(self.param_groups):
            route = self._routing[group_idx]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            adam_betas = group["adam_betas"]
            lr_adjust = group["lr_adjust"]
            lr_adjust_coeff = group["lr_adjust_coeff"]
            enable_gram = bool(group.get("enable_gram", True))
            magma_muon = bool(group.get("magma_muon", True))
            lr_device = group["lr_device"]
            adam_lr_factor = 1.0 if lr_adjust <= 0 else 1.0 / lr_adjust

            def read_grad(p: torch.Tensor) -> torch.Tensor | None:
                if grad_map is not None:
                    return grad_map.get(id(p))
                return p.grad

            # === Step 1. Collect Adam and AdamW routes ===
            adam_params: list[torch.Tensor] = []
            adam_decay_params: list[torch.Tensor] = []
            adam_grads_fp32: list[torch.Tensor] = []
            adam_exp_avgs: list[torch.Tensor] = []
            adam_exp_avg_sqs: list[torch.Tensor] = []
            adam_bias_correction1: list[torch.Tensor] = []
            adam_bias_correction2: list[torch.Tensor] = []

            for entries, decay in (
                (route["adam_no_decay"], False),
                (route["adam_decay"], True),
            ):
                for entry in entries:
                    p = entry["param"]
                    grad = read_grad(p)
                    if grad is None:
                        continue

                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                        if self._per_parameter_adam_clock:
                            state["bias_correction1"] = torch.zeros(
                                (), dtype=torch.float32, device=p.device
                            )
                            state["bias_correction2"] = torch.zeros(
                                (), dtype=torch.float32, device=p.device
                            )

                    adam_params.append(p)
                    adam_grads_fp32.append(grad.float())
                    adam_exp_avgs.append(state["exp_avg"])
                    adam_exp_avg_sqs.append(state["exp_avg_sq"])
                    if decay:
                        adam_decay_params.append(p)
                    if self._per_parameter_adam_clock:
                        if (
                            "bias_correction1" not in state
                            or "bias_correction2" not in state
                        ):
                            raise RuntimeError(
                                "HybridMuon Adam state is missing its dynamic clock"
                            )
                        adam_bias_correction1.append(state["bias_correction1"])
                        adam_bias_correction2.append(state["bias_correction2"])

            # === Step 2. Apply the fused Adam update ===
            if adam_params:
                if weight_decay > 0 and adam_decay_params:
                    self._weight_decay_inplace(
                        adam_decay_params,
                        1.0 - lr_device * (adam_lr_factor * weight_decay),
                    )

                if self._per_parameter_adam_clock:
                    self._update_adam_bias_corrections(
                        adam_bias_correction1,
                        adam_bias_correction2,
                        adam_betas[0],
                        adam_betas[1],
                    )
                else:
                    group["beta1_pow_device"].mul_(adam_betas[0])
                    group["beta2_pow_device"].mul_(adam_betas[1])

                self._adam_update_moments(
                    adam_exp_avgs,
                    adam_exp_avg_sqs,
                    adam_grads_fp32,
                    adam_betas[0],
                    adam_betas[1],
                )
                self._adam_apply_updates(
                    adam_params,
                    adam_exp_avgs,
                    adam_exp_avg_sqs,
                    adam_bias_correction1,
                    adam_bias_correction2,
                    group,
                    adam_lr_factor,
                )

            # === Step 3. Muon update for matrix parameters ===
            # === Step 3.1. Collect gradients and initialize momentum ===
            muon_params_for_decay: list[torch.Tensor] = []
            muon_grads: list[torch.Tensor] = []
            muon_momentum_buffers: list[torch.Tensor] = []
            active_entries: list[tuple[dict[str, Any], torch.Tensor]] = []

            for entry in route["muon_params"]:
                p = entry["param"]
                grad = read_grad(p)
                if grad is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                if grad.dtype != buf.dtype:
                    grad = grad.to(dtype=buf.dtype)

                muon_params_for_decay.append(p)
                muon_grads.append(grad)
                muon_momentum_buffers.append(buf)
                active_entries.append((entry, grad))

            # === Step 3.2. Apply weight decay on Muon path ===
            if weight_decay > 0 and muon_params_for_decay:
                self._weight_decay_inplace(
                    muon_params_for_decay, 1.0 - lr_device * weight_decay
                )

            if not active_entries:
                continue

            # === Step 3.3. Momentum update (Nesterov) ===
            muon_updates = _compute_muon_nesterov_updates(
                gradients=muon_grads,
                momentum_buffers=muon_momentum_buffers,
                momentum=momentum,
                use_foreach=self._use_foreach,
            )

            # === Step 3.4. Bucket by (rows, cols, device, dtype) ===
            # Merging across batch_sizes: entries with the same matrix
            # shape are concatenated along the batch dimension, producing
            # fewer but larger NS orth calls → better GPU occupancy.
            buckets: dict[
                tuple[int, int, torch.device, torch.dtype],
                list[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]],
            ] = {}

            for idx, entry_info in enumerate(active_entries):
                entry, _ = entry_info
                update_tensor = muon_updates[idx]
                bucket_key = (
                    entry["rows"],
                    entry["cols"],
                    update_tensor.device,
                    update_tensor.dtype,
                )
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(
                    (
                        entry,
                        muon_updates[idx],
                        muon_grads[idx],
                        muon_momentum_buffers[idx],
                    )
                )

            for bucket_entries in buckets.values():
                bucket_entries.sort(key=lambda item: item[0]["param"].data_ptr())

            # === Step 3.5. Pre-compute all Magma scales before NS loop ===
            # All Magma GPU kernels are launched first as a contiguous batch,
            # then all NS orth kernels follow.  This avoids interleaving
            # Magma and NS dispatches, giving the GPU a denser pipeline.
            magma_scales_map: dict[int, torch.Tensor] = {}
            if magma_muon:
                for (rows, cols, _, _), bucket_entries in buckets.items():
                    per_bucket = self._compute_magma_scales_merged(
                        bucket_entries=bucket_entries,
                        rows=rows,
                        cols=cols,
                    )
                    for (entry, _, _, _), sc in zip(
                        bucket_entries, per_bucket, strict=True
                    ):
                        magma_scales_map[id(entry["param"])] = sc

            # === Step 3.6. Newton-Schulz orthogonalization and update ===
            # Split buckets into square (standard NS) and rectangular (Gram NS).
            # Rectangular buckets are column-pad merged by min_dim in a single
            # Gram NS call per group; square buckets use the standard bmm path.
            square_buckets: dict[
                tuple[int, int, torch.device, torch.dtype],
                list[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]],
            ] = {}
            gram_buckets: dict[
                tuple[int, int, torch.device, torch.dtype],
                list[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]],
            ] = {}
            for key, bucket_entries in buckets.items():
                rows, cols = key[0], key[1]
                if enable_gram and rows != cols:
                    gram_buckets[key] = bucket_entries
                else:
                    square_buckets[key] = bucket_entries

            # The per-entry deltas of both NS paths are collected and applied
            # as one multi-tensor update after the buckets finish.
            muon_apply_params: list[torch.Tensor] = []
            muon_apply_deltas: list[torch.Tensor] = []

            # --- 3.6a  Rectangular buckets → column-pad merged Gram NS ---
            if gram_buckets:
                self._process_merged_gram_buckets(
                    gram_buckets=gram_buckets,
                    lr_adjust=lr_adjust,
                    lr_adjust_coeff=lr_adjust_coeff,
                    magma_scales_map=magma_scales_map,
                    out_params=muon_apply_params,
                    out_deltas=muon_apply_deltas,
                )

            # --- 3.6b  Square buckets → standard / flash NS path ---
            for (rows, cols, _device, _dtype), bucket_entries in square_buckets.items():
                # scale = coeff * sqrt(max(m, n))  [match-RMS mode]
                # scale = sqrt(max(1, m/n))        [rectangular mode]
                if lr_adjust <= 0:
                    scale = lr_adjust_coeff * math.sqrt(float(max(rows, cols)))
                else:
                    scale = max(1.0, rows / cols) ** 0.5

                flat_updates: list[torch.Tensor] = []
                entry_slices: list[tuple[int, int]] = []
                offset = 0
                for entry, update_tensor, _, _ in bucket_entries:
                    bs = entry["batch_size"]
                    mat = _reshape_update_to_matrix_batch(update_tensor, bs, rows, cols)
                    flat_updates.append(mat)
                    entry_slices.append((offset, bs))
                    offset += bs
                all_updates = torch.cat(flat_updates, dim=0)

                # Flash path: triton-accelerated symmetric matmul for single
                # large matrices (min_dim >= FLASH_MIN_DIM).
                total_batch = all_updates.size(0)
                M = min(rows, cols)
                use_flash = (
                    not enable_gram
                    and total_batch == 1
                    and self._use_flash
                    and _device.type == "cuda"
                    and M >= FLASH_MIN_DIM
                )
                if use_flash:
                    buf1, buf2 = self._get_ns_buffers(M, _device)
                    orthogonalized = _flash_newton_schulz_orth(
                        all_updates.squeeze(0), buf1, buf2
                    ).unsqueeze(0)
                else:
                    orthogonalized = _batched_newton_schulz_orth(all_updates)

                orthogonalized.mul_(scale)

                for idx, (entry, _, _, _) in enumerate(bucket_entries):
                    off, bs = entry_slices[idx]
                    orth_slice = orthogonalized[off : off + bs]
                    magma_scale = magma_scales_map.get(id(entry["param"]))
                    if magma_scale is not None:
                        orth_slice = orth_slice * magma_scale.view(bs, 1, 1).to(
                            dtype=orth_slice.dtype,
                            device=orth_slice.device,
                        )
                    muon_apply_params.append(entry["param"])
                    muon_apply_deltas.append(orth_slice.reshape(entry["param"].shape))

            self._apply_param_deltas(muon_apply_params, muon_apply_deltas, lr_device)
