# SPDX-License-Identifier: LGPL-3.0-or-later
"""
HybridMuon optimizer for DeePMD-kit PyTorch backend.

HybridMuon is a HYBRID optimizer that automatically combines Muon and Adam:
- For >=2D parameters with min(m,n) >= min_2d_dim: Muon update with Newton-Schulz
- For 2D parameters with min(m,n) < min_2d_dim: Adam fallback with update clipping
- For 1D parameters (biases, layer norms): Standard Adam

This is different from PyTorch's torch.optim.Muon, which ONLY supports 2D parameters
and requires manual configuration of AdamW for 1D parameters. HybridMuon provides
automatic routing based on parameter dimensionality.

Algorithm
---------
For >=2D parameters (weight matrices), the Muon update is:

    1. Momentum update (Nesterov):
       m_t = beta * m_{t-1} + (1 - beta) * g_t
       update = beta * m_t + (1 - beta) * g_t

    2. Newton-Schulz orthogonalization (quintic iteration):
       X_0 = G / ||G||_F
       X_{k+1} = a*X_k + (b*A_k + c*A_k^2) @ X_k,  where A_k = X_k @ X_k^T
       Coefficients: a=3.4445, b=-4.7750, c=2.0315

    3. Scaling: scale = coeff * sqrt(max(m, n))  [match-RMS mode]
                scale = sqrt(max(1, m/n))        [rectangular mode]

    4. Parameter update: theta -= lr * scale * orth(update)

For 1D parameters (biases, norms), standard Adam is used.

Dtype Behavior
--------------
- Newton-Schulz iterations: always bfloat16 (matches official Muon)
- NS output (bfloat16) directly applied to parameters (PyTorch handles mixed precision)
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
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from torch.optim.optimizer import (
    Optimizer,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )

# ============================================================================
# Triton availability detection
# ============================================================================

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================

# Newton-Schulz iteration count
NS_STEPS: int = 5
# Numerical stability epsilon for norm clamping and Adam
EPS: float = 1e-7
# Quintic Newton-Schulz polynomial coefficients
NS_COEFF_A: float = 3.4445
NS_COEFF_B: float = -4.7750
NS_COEFF_C: float = 2.0315
# Minimum matrix dimension for flash path to be beneficial.
# Below this threshold, triton kernel launch overhead dominates over compute,
# and cuBLAS (via torch.mm/addmm) is faster for small matrices.
FLASH_MIN_DIM: int = 1024


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
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
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
    Orthogonalize a 2D matrix via quintic Newton-Schulz with triton-accelerated
    symmetric matmul. Mathematically equivalent to ``_newton_schulz_orth``.

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
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=EPS)

    # === Step 3. Newton-Schulz iterations with triton symmetric matmul ===
    for _ in range(NS_STEPS):
        _matmul_transpose_assign(X, buf1)  # buf1 = X @ X.T = A
        _matmul_transpose_assign(buf1, buf2)  # buf2 = A @ A.T = A² (A symmetric)
        B = NS_COEFF_B * buf1 + NS_COEFF_C * buf2
        X = NS_COEFF_A * X + B @ X

    # === Step 4. Transpose back if needed ===
    if transposed:
        X = X.transpose(-2, -1)

    return X


def _newton_schulz_orth(
    G: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonalize a 2D matrix via quintic Newton-Schulz iteration.

    Mathematical formulation:
        X_0 = G / ||G||_F
        X_{k+1} = a*X_k + (b*A_k + c*A_k^2) @ X_k,  where A_k = X_k @ X_k^T
        Coefficients: a=3.4445, b=-4.7750, c=2.0315
    """
    # === Step 1. Cast to bf16 and transpose tall matrices ===
    X = G.to(dtype=torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    # === Step 2. Normalize Frobenius norm to at most 1 ===
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=EPS)

    # === Step 3. Newton-Schulz iterations with fused GEMM ===
    for _ in range(NS_STEPS):
        A = torch.mm(X, X.transpose(-2, -1))
        gram_update = torch.addmm(A, A, A, beta=NS_COEFF_B, alpha=NS_COEFF_C)
        X = torch.addmm(X, gram_update, X, beta=NS_COEFF_A, alpha=1.0)

    # === Step 4. Transpose back if needed ===
    if transposed:
        X = X.transpose(-2, -1)

    return X


def should_fallback_to_adam_for_matrix(
    p: torch.Tensor,
    min_2d_dim: int,
) -> bool:
    """
    Check if a 2D matrix should fallback to Adam due to small dimensions.

    Parameters
    ----------
    p : torch.Tensor
        Parameter tensor with ndim >= 2.
    min_2d_dim : int
        Minimum min(m, n) threshold for Muon. Matrices with min(m, n) >=
        min_2d_dim use Muon; those with min(m, n) < min_2d_dim use Adam.

    Returns
    -------
    bool
        True if min(m, n) < min_2d_dim, False otherwise.

    Raises
    ------
    ValueError
        If tensor has ndim < 2.
    """
    # === Step 1. Validate ===
    if p.ndim < 2:
        raise ValueError("Parameter must have ndim >= 2 for Muon suitability check.")

    # === Step 2. Derive matrix shape consistent with Muon reshape ===
    m = int(p.shape[0])
    n = int(p.numel() // p.shape[0])

    # === Step 3. Check if any dimension too small for Muon ===
    return min(m, n) < min_2d_dim


class HybridMuonOptimizer(Optimizer):
    """
    HybridMuon optimizer with small-2D Adam fallback and 1D Adam path.

    This optimizer applies different update rules based on parameter dimensionality:
    - For >=2D parameters with min(m, n) >= min_2d_dim:
      Muon update with Newton-Schulz orthogonalization.
    - For 2D parameters with min(m, n) < min_2d_dim (small matrices):
      Adam update with scaled learning rate and update clipping.
    - For 1D parameters (biases, layer norms):
      Standard Adam update.

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

    Adam (1D params):
        Standard Adam with bias correction, all computations in float32.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize.
    lr : float
        Learning rate with default 1e-3.
    momentum : float
        Momentum coefficient for Muon with default 0.95.
    weight_decay : float
        Weight decay coefficient (applied only to Muon-routed parameters) with default 0.001.
    adam_betas : tuple[float, float]
        Adam beta coefficients with default (0.9, 0.95).
    lr_adjust : float
        Learning rate adjustment mode for Muon scaling and Adam learning rate.
        - If lr_adjust <= 0: use match-RMS scaling for Muon,
          scale = lr_adjust_coeff * sqrt(max(m, n)). Adam uses lr directly.
        - If lr_adjust > 0: use rectangular correction for Muon,
          scale = sqrt(max(1.0, m/n)). Adam uses lr/lr_adjust.
        Default is 10.0 (Adam lr = lr/10).
    lr_adjust_coeff : float
        Dual-purpose coefficient with default 0.2:
        1. For Muon (when lr_adjust <= 0): match-RMS scaling factor,
           scale = lr_adjust_coeff * sqrt(max(m, n)).
        2. For 2D Adam fallback: learning rate multiplier,
           adam_lr_matrix = adam_lr * min(lr_adjust_coeff, 0.1).
           The min(., 0.1) cap ensures conservative updates for small matrices.
    muon_2d_only : bool
        If True, only 2D parameters use Muon (matching PyTorch's torch.optim.Muon).
        Parameters with ndim > 2 use Adam without weight decay.
        If False, all >=2D parameters use Muon (default behavior).
        Default is True.
    min_2d_dim : int
        Minimum min(m, n) threshold for Muon on 2D matrices.
        Matrices with min(m, n) >= min_2d_dim use Muon;
        those with min(m, n) < min_2d_dim use Adam fallback.
        Must be >= 1.
        Set to 1 to disable fallback.
        Default is 1.
    flash_muon : bool
        Enable triton-accelerated Newton-Schulz orthogonalization.
        Requires triton and CUDA. Falls back to PyTorch implementation
        when triton is unavailable or running on CPU.
        Default is True.

    Examples
    --------
    >>> optimizer = HybridMuonOptimizer(model.parameters(), lr=1e-3)
    >>> for epoch in range(epochs):
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.001,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        lr_adjust: float = 10.0,
        lr_adjust_coeff: float = 0.2,
        muon_2d_only: bool = True,
        min_2d_dim: int = 1,
        flash_muon: bool = True,
    ) -> None:
        if min_2d_dim < 1:
            raise ValueError("min_2d_dim must be >= 1.")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "adam_betas": adam_betas,
            "lr_adjust": lr_adjust,
            "lr_adjust_coeff": lr_adjust_coeff,
            "muon_2d_only": muon_2d_only,
            "min_2d_dim": min_2d_dim,
        }
        super().__init__(params, defaults)
        # Static parameter routing: built once on first step() call.
        self._routing_built = False
        self._routing: list[dict[str, Any]] = []

        # Flash-Muon: triton-accelerated Newton-Schulz
        self._use_flash = flash_muon and TRITON_AVAILABLE
        # Lazily allocated NS iteration buffers, keyed by (M, device)
        self._ns_buffers: dict[
            tuple[int, torch.device],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

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

    def _build_param_routing(self) -> None:
        """
        Classify parameters into Muon and Adam routes (static routing).

        Routing logic:
        - 1D parameters → Adam path
        - >2D parameters (when muon_2d_only=True) → Adam path
        - 2D parameters with min(m, n) < min_2d_dim → Adam fallback path
        - 2D parameters with min(m, n) >= min_2d_dim → Muon path
        - >=2D parameters (when muon_2d_only=False) → Muon path
        """
        if self._routing_built:
            return

        self._routing = []
        for group in self.param_groups:
            muon_params: list[dict[str, Any]] = []
            adam_1d: list[dict[str, Any]] = []
            adam_matrix: list[dict[str, Any]] = []
            adam_nd: list[dict[str, Any]] = []

            min_2d_dim = group["min_2d_dim"]
            muon_2d_only = group["muon_2d_only"]

            for p in group["params"]:
                # === Step 1. 1D parameters → Adam ===
                if p.ndim < 2:
                    adam_1d.append({"param": p})
                    continue

                # === Step 2. >2D parameters (when muon_2d_only=True) → Adam ===
                if muon_2d_only and p.ndim > 2:
                    adam_nd.append({"param": p})
                    continue

                # === Step 3. 2D small matrices → Adam fallback ===
                if (p.ndim == 2) and should_fallback_to_adam_for_matrix(
                    p, min_2d_dim=min_2d_dim
                ):
                    adam_matrix.append(
                        {
                            "param": p,
                            "abs_floor": 1e-3 * math.sqrt(float(p.numel())),
                        }
                    )
                    continue

                # === Step 4. >=2D (or 2D only when muon_2d_only=True) → Muon ===
                muon_params.append(
                    {
                        "param": p,
                        "rows": int(p.shape[0]),
                        "cols": int(p.numel() // p.shape[0]),
                    }
                )

            self._routing.append(
                {
                    "muon_params": muon_params,
                    "adam_1d": adam_1d,
                    "adam_matrix": adam_matrix,
                    "adam_nd": adam_nd,
                }
            )

        self._routing_built = True

    @torch.no_grad()
    def step(
        self,
        closure: callable | None = None,
    ) -> torch.Tensor | None:
        """
        Perform a single optimization step.

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

        for group_idx, group in enumerate(self.param_groups):
            route = self._routing[group_idx]
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            adam_betas = group["adam_betas"]
            lr_adjust = group["lr_adjust"]
            lr_adjust_coeff = group["lr_adjust_coeff"]

            # === Step 1. Adam update for 1D parameters (biases, norms, etc.) ===
            # === Step 1.1. Collect gradients and initialize state ===
            adam_params: list[torch.Tensor] = []
            adam_grads_fp32: list[torch.Tensor] = []
            adam_exp_avgs: list[torch.Tensor] = []
            adam_exp_avg_sqs: list[torch.Tensor] = []
            adam_states: list[dict[str, Any]] = []

            for entry in route["adam_1d"]:
                p = entry["param"]
                grad = p.grad
                if grad is None:
                    continue

                grad_fp32 = grad.float()

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["beta1_pow"] = 1.0
                    state["beta2_pow"] = 1.0

                state["beta1_pow"] *= adam_betas[0]
                state["beta2_pow"] *= adam_betas[1]

                adam_params.append(p)
                adam_grads_fp32.append(grad_fp32)
                adam_exp_avgs.append(state["exp_avg"])
                adam_exp_avg_sqs.append(state["exp_avg_sq"])
                adam_states.append(state)

            if adam_params:
                # === Step 1.2. Update exp_avg / exp_avg_sq ===
                adam_lr = lr if lr_adjust <= 0 else lr / lr_adjust

                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                for ea, g in zip(adam_exp_avgs, adam_grads_fp32):
                    ea.lerp_(g, 1 - adam_betas[0])
                grad_sq = [g * g for g in adam_grads_fp32]
                for eas, gsq in zip(adam_exp_avg_sqs, grad_sq):
                    eas.lerp_(gsq, 1 - adam_betas[1])

                # === Step 1.3. Bias correction and parameter update ===
                for i, p in enumerate(adam_params):
                    state = adam_states[i]
                    bias_corr1 = 1 - state["beta1_pow"]
                    bias_corr2 = 1 - state["beta2_pow"]
                    step_size = adam_lr / bias_corr1
                    # delta = -step_size * m_hat / (sqrt(v_hat) + eps)
                    denom = (adam_exp_avg_sqs[i] / bias_corr2).sqrt().add_(EPS)
                    delta_fp32 = -step_size * (adam_exp_avgs[i] / denom)
                    p.add_(delta_fp32.to(p.dtype))

            # === Step 2. Adam update for >2D parameters (when muon_2d_only=True) ===
            # === Step 2.1. Collect gradients and initialize state ===
            adam_nd_params: list[torch.Tensor] = []
            adam_nd_grads_fp32: list[torch.Tensor] = []
            adam_nd_exp_avgs: list[torch.Tensor] = []
            adam_nd_exp_avg_sqs: list[torch.Tensor] = []
            adam_nd_states: list[dict[str, Any]] = []

            for entry in route.get("adam_nd", []):
                p = entry["param"]
                grad = p.grad
                if grad is None:
                    continue

                grad_fp32 = grad.float()

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["beta1_pow"] = 1.0
                    state["beta2_pow"] = 1.0

                state["beta1_pow"] *= adam_betas[0]
                state["beta2_pow"] *= adam_betas[1]

                adam_nd_params.append(p)
                adam_nd_grads_fp32.append(grad_fp32)
                adam_nd_exp_avgs.append(state["exp_avg"])
                adam_nd_exp_avg_sqs.append(state["exp_avg_sq"])
                adam_nd_states.append(state)

            if adam_nd_params:
                # === Step 2.2. Update exp_avg / exp_avg_sq ===
                adam_lr = lr if lr_adjust <= 0 else lr / lr_adjust

                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                for ea, g in zip(adam_nd_exp_avgs, adam_nd_grads_fp32):
                    ea.lerp_(g, 1 - adam_betas[0])
                grad_sq = [g * g for g in adam_nd_grads_fp32]
                for eas, gsq in zip(adam_nd_exp_avg_sqs, grad_sq):
                    eas.lerp_(gsq, 1 - adam_betas[1])

                # === Step 2.3. Bias correction and parameter update ===
                for i, p in enumerate(adam_nd_params):
                    state = adam_nd_states[i]
                    bias_corr1 = 1 - state["beta1_pow"]
                    bias_corr2 = 1 - state["beta2_pow"]
                    step_size = adam_lr / bias_corr1
                    # delta = -step_size * m_hat / (sqrt(v_hat) + eps)
                    denom = (adam_nd_exp_avg_sqs[i] / bias_corr2).sqrt().add_(EPS)
                    delta_fp32 = -step_size * (adam_nd_exp_avgs[i] / denom)
                    p.add_(delta_fp32.to(p.dtype))

            # === Step 3. Adam update for small 2D matrices (fallback path) ===
            # === Step 3.1. Collect gradients and initialize state ===
            adam_matrix_params: list[torch.Tensor] = []
            adam_matrix_grads_fp32: list[torch.Tensor] = []
            adam_matrix_exp_avgs: list[torch.Tensor] = []
            adam_matrix_exp_avg_sqs: list[torch.Tensor] = []
            adam_matrix_states: list[dict[str, Any]] = []
            adam_matrix_abs_floor: list[float] = []

            for entry in route["adam_matrix"]:
                p = entry["param"]
                grad = p.grad
                if grad is None:
                    continue

                grad_fp32 = grad.float()

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["beta1_pow"] = 1.0
                    state["beta2_pow"] = 1.0

                state["beta1_pow"] *= adam_betas[0]
                state["beta2_pow"] *= adam_betas[1]

                adam_matrix_params.append(p)
                adam_matrix_grads_fp32.append(grad_fp32)
                adam_matrix_exp_avgs.append(state["exp_avg"])
                adam_matrix_exp_avg_sqs.append(state["exp_avg_sq"])
                adam_matrix_states.append(state)
                adam_matrix_abs_floor.append(entry["abs_floor"])

            if adam_matrix_params:
                # === Step 3.2. Update exp_avg / exp_avg_sq with scaled lr ===
                adam_lr = lr if lr_adjust <= 0 else lr / lr_adjust
                adam_lr_matrix = adam_lr * min(lr_adjust_coeff, 0.1)

                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                for ea, g in zip(adam_matrix_exp_avgs, adam_matrix_grads_fp32):
                    ea.lerp_(g, 1 - adam_betas[0])
                grad_sq_m = [g * g for g in adam_matrix_grads_fp32]
                for eas, gsq in zip(adam_matrix_exp_avg_sqs, grad_sq_m):
                    eas.lerp_(gsq, 1 - adam_betas[1])

                # === Step 3.3. Compute unclipped deltas ===
                raw_deltas: list[torch.Tensor] = []
                for i in range(len(adam_matrix_params)):
                    state = adam_matrix_states[i]
                    bias_corr1 = 1 - state["beta1_pow"]
                    bias_corr2 = 1 - state["beta2_pow"]
                    step_size = adam_lr_matrix / bias_corr1
                    denom = (adam_matrix_exp_avg_sqs[i] / bias_corr2).sqrt().add_(EPS)
                    raw_deltas.append(-step_size * (adam_matrix_exp_avgs[i] / denom))

                # === Step 3.4. Clip updates by relative norm and apply ===
                max_rel_change = 0.05
                p_norms = torch.stack([p.norm() for p in adam_matrix_params])
                delta_norms = torch.stack([d.norm() for d in raw_deltas])
                floors = torch.tensor(
                    adam_matrix_abs_floor,
                    device=p_norms.device,
                    dtype=p_norms.dtype,
                )
                max_delta = torch.maximum(max_rel_change * p_norms, floors)
                scales_tensor = torch.clamp(max_delta / (delta_norms + 1e-12), max=1.0)
                for i, (p, delta) in enumerate(
                    zip(adam_matrix_params, raw_deltas, strict=False)
                ):
                    p.add_(delta.mul_(scales_tensor[i]).to(p.dtype))

            # === Step 4. Muon update for >=2D parameters (weight matrices) ===
            # === Step 4.1. Collect gradients and initialize momentum ===
            muon_params_for_decay: list[torch.Tensor] = []
            muon_grads: list[torch.Tensor] = []
            muon_momentum_buffers: list[torch.Tensor] = []
            active_entries: list[tuple[dict[str, Any], torch.Tensor]] = []

            for entry in route["muon_params"]:
                p = entry["param"]
                grad = p.grad
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

            # === Step 4.2. Apply weight decay (Muon path only) ===
            if weight_decay > 0 and muon_params_for_decay:
                for p in muon_params_for_decay:
                    p.mul_(1.0 - lr * weight_decay)

            if not active_entries:
                continue

            # === Step 4.3. Momentum update (Nesterov) ===
            # m_t = beta * m_{t-1} + (1 - beta) * g_t
            for buf, g in zip(muon_momentum_buffers, muon_grads):
                buf.lerp_(g, 1 - momentum)
            # update = beta * m_t + (1 - beta) * g_t
            muon_updates = [
                torch.lerp(g, buf, momentum)
                for g, buf in zip(muon_grads, muon_momentum_buffers)
            ]

            # === Step 4.4. Bucket by shape/device/dtype for batched NS ===
            buckets: dict[
                tuple[int, int, torch.device, torch.dtype],
                list[tuple[dict[str, Any], torch.Tensor]],
            ] = {}

            for idx, entry_info in enumerate(active_entries):
                entry, _ = entry_info
                p = entry["param"]
                bucket_key = (entry["rows"], entry["cols"], p.device, p.dtype)
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append((entry, muon_updates[idx]))

            # === Step 4.5. Newton-Schulz orthogonalization and update ===
            for (rows, cols, _device, _), bucket_entries in buckets.items():
                # scale = coeff * sqrt(max(m, n))  [match-RMS mode]
                # scale = sqrt(max(1, m/n))        [rectangular mode]
                if lr_adjust <= 0:
                    scale = lr_adjust_coeff * math.sqrt(float(max(rows, cols)))
                else:
                    scale = max(1.0, rows / cols) ** 0.5

                # Determine if flash path is usable for this bucket.
                # Only beneficial when min(rows, cols) >= FLASH_MIN_DIM;
                # for small matrices, triton launch overhead > compute savings.
                M = min(rows, cols)
                use_flash = (
                    self._use_flash and _device.type == "cuda" and M >= FLASH_MIN_DIM
                )
                if use_flash:
                    buf1, buf2 = self._get_ns_buffers(M, _device)

                # Process each entry individually with Newton-Schulz orth.
                # Compatible with sharding propagation under FSDP2.
                for entry, update_tensor in bucket_entries:
                    update_matrix = update_tensor.reshape(rows, cols)
                    if not update_matrix.is_contiguous():
                        update_matrix = update_matrix.contiguous()

                    if use_flash:
                        orth = _flash_newton_schulz_orth(update_matrix, buf1, buf2)
                    else:
                        orth = _newton_schulz_orth(update_matrix)
                    orth.mul_(scale)
                    delta = orth.reshape(entry["param"].shape)
                    entry["param"].add_(delta, alpha=-lr)

        return loss
