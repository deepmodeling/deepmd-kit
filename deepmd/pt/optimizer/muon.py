# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Muon optimizer for DeePMD-kit PyTorch backend.

Muon is an optimizer that applies Newton-Schulz orthogonalization to the gradient
before using momentum, resulting in orthogonalized updates for weight matrices.
This can improve training stability and convergence for certain architectures.

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
- Adam state (exp_avg, exp_avg_sq): always float32 for numerical stability
- Gradients: cast to parameter dtype before momentum update

Reference
---------
https://github.com/KellerJordan/Muon
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
# Constants
# ============================================================================

# Newton-Schulz iteration count
NS_STEPS: int = 5
# Numerical stability epsilon for norm clamping
NS_EPS: float = 1e-7
# Adam epsilon for numerical stability
ADAM_EPS: float = 1e-7


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the zeroth power (orthogonalization) of a matrix via Newton-Schulz iteration.

    Uses quintic Newton-Schulz iteration to compute the orthogonal component of the
    input matrix. This is equivalent to computing U from the SVD decomposition G = USV^T.

    Mathematical formulation:
        X_0 = G / ||G||_F
        X_{k+1} = a*X_k + (b*A_k + c*A_k^2) @ X_k,  where A_k = X_k @ X_k^T
        Coefficients: a=3.4445, b=-4.7750, c=2.0315

    This implementation matches PyTorch official Muon behavior: it always performs
    Newton-Schulz in bfloat16 and returns a bfloat16 tensor.

    Parameters
    ----------
    G : torch.Tensor
        Input matrix to orthogonalize with shape (..., M, N).

    Returns
    -------
    torch.Tensor
        Orthogonalized matrix in bfloat16 with same shape as input.

    Raises
    ------
    ValueError
        If G has fewer than 2 dimensions.
    """
    # === Step 1. Validate ===
    if G.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions (..., M, N).")

    a, b, c = (3.4445, -4.7750, 2.0315)

    # === Step 2. Cast to bf16 (match official Muon) ===
    X = G.to(dtype=torch.bfloat16)

    # === Step 3. Transpose tall matrices ===
    if X.size(-2) > X.size(-1):
        X = X.mT

    # === Step 4. Normalize Frobenius norm to at most 1 ===
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=NS_EPS)

    # === Step 5. Newton-Schulz iterations with fused GEMM ===
    for _ in range(NS_STEPS):
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
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Prepare momentum update and reshape for batched Newton-Schulz.

    Uses Nesterov momentum: update = beta*m_t + (1-beta)*g_t, where m_t is
    the updated momentum buffer.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient tensor.
    momentum_buffer : torch.Tensor
        Momentum buffer (will be updated in-place).
    beta : float
        Momentum coefficient.

    Returns
    -------
    update : torch.Tensor
        Reshaped update tensor with shape (M, N).
    original_shape : tuple[int, ...]
        Original shape before reshape.
    """
    # === Step 1. Update momentum buffer ===
    momentum_buffer.lerp_(grad, 1 - beta)
    # Nesterov lookahead
    update = grad.lerp(momentum_buffer, beta)

    # === Step 2. Handle tensor -> matrix reshape ===
    original_shape = update.shape
    if update.ndim > 2:
        update = update.reshape(update.shape[0], -1)

    return update, original_shape


class MuonOptimizer(Optimizer):
    """
    Muon optimizer with auxiliary Adam for non-matrix parameters.

    This optimizer applies different update rules based on parameter dimensionality:
    - For >=2D parameters (weight matrices): Muon update with Newton-Schulz orthogonalization
    - For 1D parameters (biases, layer norms): Standard Adam update

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
        Weight decay coefficient (applied only to >=2D params) with default 0.001.
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
        Coefficient for match-RMS scaling with default 0.2.
        Only effective when lr_adjust <= 0.

    Examples
    --------
    >>> optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
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
        lr_adjust: float = 0.0,
        lr_adjust_coeff: float = 0.2,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "adam_betas": adam_betas,
            "lr_adjust": lr_adjust,
            "lr_adjust_coeff": lr_adjust_coeff,
        }
        super().__init__(params, defaults)

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

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            adam_betas = group["adam_betas"]
            lr_adjust = group["lr_adjust"]
            lr_adjust_coeff = group["lr_adjust_coeff"]

            # === Step 1. Collect params with gradients and separate by type ===
            muon_params: list[torch.Tensor] = []  # For weight decay (>=2D only)
            muon_entries: list[tuple[torch.nn.Parameter, torch.Tensor, tuple]] = []
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
                    # Muon path: collect for weight decay
                    muon_params.append(p)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    update, orig_shape = _prepare_muon_momentum(
                        grad, state["momentum_buffer"], momentum
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
                    denom = (adam_exp_avg_sqs[i] / bias_corr2).sqrt().add_(ADAM_EPS)
                    delta_fp32 = -step_size * (adam_exp_avgs[i] / denom)
                    p.add_(delta_fp32.to(p.dtype))

            # === Step 4. Batched Newton-Schulz for Muon parameters ===
            if not muon_entries:
                continue

            # Group by (rows, cols, device) for batched processing
            # Note: dtype is not included since NS internally converts to bf16
            buckets: dict[
                tuple[int, int, torch.device],
                list[tuple[torch.nn.Parameter, torch.Tensor, tuple]],
            ] = {}
            for entry in muon_entries:
                p, update, orig_shape = entry
                key = (update.shape[0], update.shape[1], update.device)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(entry)

            # Process each bucket
            for bucket in buckets.values():
                # === Pre-compute bucket-level scaling constants ===
                # Get matrix dimensions from first entry
                m, n = bucket[0][1].shape[-2], bucket[0][1].shape[-1]
                # Scaling: match-RMS (lr_adjust<=0) or rectangular correction
                if lr_adjust <= 0:
                    scale = lr_adjust_coeff * math.sqrt(float(max(m, n)))
                else:
                    scale = max(1.0, m / n) ** 0.5

                # === Stack and orthogonalize ===
                if len(bucket) == 1:
                    # Single parameter: 2D path with addmm (faster, correct behavior)
                    p, update, orig_shape = bucket[0]
                    orth = zeropower_via_newtonschulz5(update)
                    # === Apply scaling and update parameters ===
                    orth.mul_(scale)
                    p.add_(orth.reshape(orig_shape), alpha=-lr)
                else:
                    # Multiple parameters: 3D batched path with baddbmm
                    stacked = torch.stack(
                        [item[1].contiguous() for item in bucket], dim=0
                    )
                    orth_stacked = zeropower_via_newtonschulz5(stacked)
                    # === Apply scaling and update parameters ===
                    orth_stacked.mul_(scale)
                    for i, (p, _, orig_shape) in enumerate(bucket):
                        p.add_(orth_stacked[i].reshape(orig_shape), alpha=-lr)

        return loss
