# SPDX-License-Identifier: LGPL-3.0-or-later
"""
AdaMuon optimizer for DeePMD-kit PyTorch backend.

AdaMuon combines Newton-Schulz orthogonalization with adaptive per-element
second-moment normalization and RMS-aligned global scaling. It applies sign-stabilized
orthogonal direction for improved training stability.

Key improvements over vanilla Muon:
- Sign-stabilized orthogonal direction
- Per-element second-moment (v_buffer) normalization
- RMS-aligned global scaling

Reference:
    https://github.com/ethansmith2000/AdaMuon
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
        Callable,
        Iterable,
    )


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute the zeroth power (orthogonalization) of a matrix via Newton-Schulz iteration.

    Uses quintic Newton-Schulz iteration to compute the orthogonal component of the
    input matrix. This is equivalent to computing U from the SVD decomposition G = USV^T.

    This implementation always performs Newton-Schulz in bfloat16 and returns a
    bfloat16 tensor.

    Parameters
    ----------
    G : torch.Tensor
        Input matrix to orthogonalize with shape (..., M, N).
    steps : int
        Number of Newton-Schulz iterations with default 5.
    eps : float
        Numerical stability epsilon for norm clamping with default 1e-7.

    Returns
    -------
    torch.Tensor
        Orthogonalized matrix in bfloat16 with same shape as input.

    Raises
    ------
    ValueError
        If G has fewer than 2 dimensions.
    ValueError
        If steps >= 100 (guard for efficiency).
    """
    # === Step 1. Validate ===
    if G.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions (..., M, N).")
    if steps >= 100:
        raise ValueError("Number of steps must be less than 100 for efficiency.")

    a, b, c = (3.4445, -4.7750, 2.0315)

    # === Step 2. Cast to bf16 ===
    X = G.to(dtype=torch.bfloat16)

    # === Step 3. Transpose tall matrices ===
    if X.size(-2) > X.size(-1):
        X = X.mT

    # === Step 4. Normalize Frobenius norm to at most 1 ===
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)

    # === Step 5. Newton-Schulz iterations with fused GEMM ===
    for _ in range(steps):
        A = X @ X.mT
        # gram_update = b*A + c*(A@A) via addmm/baddbmm
        # X = a*X + gram_update@X via addmm/baddbmm
        if X.ndim == 2:
            gram_update = torch.addmm(A, A, A, beta=b, alpha=c)
            X = torch.addmm(X, gram_update, X, beta=a, alpha=1.0)
        else:
            gram_update = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, gram_update, X, beta=a, alpha=1.0)

    # === Step 6. Transpose back if needed ===
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


def _prepare_muon_momentum(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    nesterov: bool,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Prepare momentum update and reshape for batched Newton-Schulz.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient tensor.
    momentum_buffer : torch.Tensor
        Momentum buffer (will be updated in-place).
    beta : float
        Momentum coefficient.
    nesterov : bool
        Whether to use Nesterov momentum.

    Returns
    -------
    update : torch.Tensor
        Reshaped update tensor with shape (M, N).
    original_shape : tuple[int, ...]
        Original shape before reshape.
    """
    # === Step 1. Update momentum buffer ===
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum_buffer, beta) if nesterov else momentum_buffer

    # === Step 2. Handle tensor -> matrix reshape ===
    original_shape = update.shape
    if update.ndim > 2:
        update = update.reshape(update.shape[0], -1)

    return update, original_shape


class AdaMuonOptimizer(Optimizer):
    """
    AdaMuon optimizer with adaptive second-moment normalization and auxiliary Adam.

    This optimizer applies different update rules based on parameter dimensionality:
    - For 2D+ parameters (weight matrices): AdaMuon update with sign-stabilized
      Newton-Schulz orthogonalization and per-element v_buffer normalization.
    - For 1D parameters (biases, layer norms): Standard Adam update.

    Key AdaMuon features:
    - Sign-stabilized orthogonal direction: Applies sign() before orthogonalization.
    - Per-element second-moment normalization using momentum coefficient.
    - RMS-aligned global scaling: 0.2 * sqrt(min * max) / norm.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize.
    lr : float
        Learning rate with default 1e-3.
    momentum : float
        Momentum coefficient for AdaMuon with default 0.95.
    weight_decay : float
        Weight decay coefficient (applied only to >=2D params) with default 0.001.
    ns_steps : int
        Number of Newton-Schulz iterations with default 5.
    adam_betas : tuple[float, float]
        Adam beta coefficients with default (0.9, 0.95).
    adam_eps : float
        Adam epsilon with default 1e-7.
    nesterov : bool
        Whether to use Nesterov momentum for AdaMuon with default True.
    lr_adjust : float
        Learning rate adjustment factor for Adam (1D params).
        - If lr_adjust <= 0: use match-RMS scaling for AdaMuon update,
          scale = lr_adjust_coeff * sqrt(max(m, n)). Adam uses lr directly.
        - If lr_adjust > 0: use rectangular correction for AdaMuon update,
          scale = sqrt(max(1.0, m/n)). Adam uses lr/lr_adjust as learning rate.
        Default is 10.0 (Adam lr = lr/10).
    lr_adjust_coeff : float
        Coefficient for match-RMS scaling with default 0.2.
        Only effective when lr_adjust <= 0.
    eps : float
        Epsilon for v_buffer sqrt and global scaling normalization with default 1e-8.

    Examples
    --------
    >>> optimizer = AdaMuonOptimizer(model.parameters(), lr=1e-3)
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
        ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-7,
        nesterov: bool = True,
        lr_adjust: float = 10.0,
        lr_adjust_coeff: float = 0.2,
        eps: float = 1e-8,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "adam_betas": adam_betas,
            "adam_eps": adam_eps,
            "nesterov": nesterov,
            "lr_adjust": lr_adjust,
            "lr_adjust_coeff": lr_adjust_coeff,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """
        Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss : float, optional
            The loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            adam_betas = group["adam_betas"]
            adam_eps = group["adam_eps"]
            nesterov = group["nesterov"]
            lr_adjust = group["lr_adjust"]
            lr_adjust_coeff = group["lr_adjust_coeff"]
            eps = group["eps"]

            # === Step 1. Collect params with gradients and separate by type ===
            muon_params: list[torch.Tensor] = []  # For weight decay (>=2D only)
            muon_entries: list[
                tuple[torch.nn.Parameter, torch.Tensor, tuple[int, ...]]
            ] = []
            # Adam batch lists
            adam_params: list[torch.Tensor] = []
            adam_grads_fp32: list[torch.Tensor] = []
            adam_exp_avgs: list[torch.Tensor] = []
            adam_exp_avg_sqs: list[torch.Tensor] = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype != p.dtype:
                    grad = grad.to(dtype=p.dtype)

                state = self.state[p]

                if p.ndim >= 2:
                    # AdaMuon path: collect for weight decay
                    muon_params.append(p)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    update, orig_shape = _prepare_muon_momentum(
                        grad, state["momentum_buffer"], momentum, nesterov
                    )
                    muon_entries.append((p, update, orig_shape))
                else:
                    # Adam path: state tensors forced to FP32
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                        state["beta1_pow"] = 1.0
                        state["beta2_pow"] = 1.0
                    state["beta1_pow"] *= adam_betas[0]
                    state["beta2_pow"] *= adam_betas[1]
                    adam_params.append(p)
                    # Cast grad to FP32 for Adam computation
                    adam_grads_fp32.append(grad.float())
                    adam_exp_avgs.append(state["exp_avg"])
                    adam_exp_avg_sqs.append(state["exp_avg_sq"])

            # === Step 2. Foreach weight decay (only >=2D params) ===
            if weight_decay > 0 and muon_params:
                torch._foreach_mul_(muon_params, 1.0 - lr * weight_decay)

            # === Step 3. Adam update for 1D params (FP32 computation) ===
            if adam_params:
                # Determine Adam learning rate
                adam_lr = lr if lr_adjust <= 0 else lr / lr_adjust

                # Update momentum estimates in FP32
                torch._foreach_lerp_(adam_exp_avgs, adam_grads_fp32, 1 - adam_betas[0])
                grad_sq = torch._foreach_mul(adam_grads_fp32, adam_grads_fp32)
                torch._foreach_lerp_(adam_exp_avg_sqs, grad_sq, 1 - adam_betas[1])

                # Compute updates with bias correction (per-param beta_pow)
                for i, p in enumerate(adam_params):
                    state = self.state[p]
                    bias_corr1 = 1 - state["beta1_pow"]
                    bias_corr2 = 1 - state["beta2_pow"]
                    step_size = adam_lr / bias_corr1
                    # FP32 computation: compute full delta in FP32, then cast once
                    denom = (adam_exp_avg_sqs[i] / bias_corr2).sqrt().add_(adam_eps)
                    delta_fp32 = -step_size * (adam_exp_avgs[i] / denom)
                    p.add_(delta_fp32.to(p.dtype))

            # === Step 4. Batched Newton-Schulz for AdaMuon parameters ===
            if not muon_entries:
                continue

            # Group by (rows, cols, device) for batched processing
            buckets: dict[
                tuple[int, int, torch.device],
                list[tuple[torch.nn.Parameter, torch.Tensor, tuple[int, ...]]],
            ] = {}
            for entry in muon_entries:
                p, update, orig_shape = entry
                key = (update.shape[0], update.shape[1], update.device)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(entry)

            # Process each bucket
            for (rows, cols, _device), bucket in buckets.items():
                m, n = rows, cols

                # === Pre-compute bucket-level scaling constants ===
                # RMS-aligned scale: 0.2 * sqrt(m * n)
                rms_scale = 0.2 * math.sqrt(float(m * n))
                # Shape-dependent lr correction (based on lr_adjust mode)
                if lr_adjust <= 0:
                    adj_scale = lr_adjust_coeff * math.sqrt(float(max(m, n)))
                else:
                    adj_scale = max(1.0, m / n) ** 0.5

                # === Step 4.1 Stack sign matrices and orthogonalize ===
                # Always stack to 3D (B, m, n) for unified indexing
                stacked = torch.stack(
                    [torch.sign(item[1].contiguous()) for item in bucket], dim=0
                )
                orth_stacked = zeropower_via_newtonschulz5(stacked, steps=ns_steps)

                # === Step 4.2 Per-element v_buffer normalization and update ===
                for i, (p, update, orig_shape) in enumerate(bucket):
                    state = self.state[p]
                    # orth_stacked is always 3D, use unified indexing
                    orth_vec = (
                        orth_stacked[i].flatten().float()
                    )  # Cast to FP32 for stability

                    # === Step 4.2.1 Initialize or retrieve v_buffer ===
                    if "v_buffer" not in state:
                        state["v_buffer"] = torch.zeros(
                            orth_vec.numel(),
                            dtype=torch.float32,
                            device=orth_vec.device,
                        )
                    v = state["v_buffer"]

                    # === Step 4.2.2 EMA update and element-wise normalization ===
                    # v = momentum * v + (1 - momentum) * orth_vec^2
                    v.mul_(momentum).addcmul_(orth_vec, orth_vec, value=1.0 - momentum)
                    orth_vec = orth_vec / (v.sqrt().add_(eps))

                    # === Step 4.2.3 RMS-aligned global scaling ===
                    # scale = rms_scale / (norm + eps)
                    norm_val = orth_vec.norm()
                    orth_vec.div_(norm_val + eps).mul_(rms_scale)

                    # === Step 4.2.4 Shape-dependent lr correction ===
                    orth_vec.mul_(adj_scale)

                    # Reshape back and update parameter
                    p.add_(
                        orth_vec.view(m, n).reshape(orig_shape).to(p.dtype), alpha=-lr
                    )

        return loss
