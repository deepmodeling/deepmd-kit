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
- Muon gradients: cast to parameter dtype before momentum update
- Adam gradients: cast to float32 for update computation

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
# Quintic Newton-Schulz polynomial coefficients
NS_COEFF_A: float = 3.4445
NS_COEFF_B: float = -4.7750
NS_COEFF_C: float = 2.0315


def _maybe_compile(
    fn: callable,
) -> callable:
    """Compile a function if torch.compile is available."""
    if hasattr(torch, "compile"):
        return torch.compile(fn, fullgraph=True, dynamic=True)
    return fn


@_maybe_compile
def _zeropower_via_newtonschulz5_2d(
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
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=NS_EPS)

    # === Step 3. Newton-Schulz iterations with fused GEMM ===
    for _ in range(NS_STEPS):
        A = torch.mm(X, X.transpose(-2, -1))
        gram_update = torch.addmm(A, A, A, beta=NS_COEFF_B, alpha=NS_COEFF_C)
        X = torch.addmm(X, gram_update, X, beta=NS_COEFF_A, alpha=1.0)

    # === Step 4. Transpose back if needed ===
    if transposed:
        X = X.transpose(-2, -1)

    return X


@_maybe_compile
def _zeropower_via_newtonschulz5_3d(
    G: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonalize a 3D batch of matrices via quintic Newton-Schulz iteration.

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
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=NS_EPS)

    # === Step 3. Newton-Schulz iterations with batched fused GEMM ===
    for _ in range(NS_STEPS):
        A = torch.bmm(X, X.transpose(-2, -1))
        gram_update = torch.baddbmm(A, A, A, beta=NS_COEFF_B, alpha=NS_COEFF_C)
        X = torch.baddbmm(X, gram_update, X, beta=NS_COEFF_A, alpha=1.0)

    # === Step 4. Transpose back if needed ===
    if transposed:
        X = X.transpose(-2, -1)

    return X


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the zeroth power (orthogonalization) via Newton-Schulz iteration.

    Dispatches to compiled 2D or 3D kernels for best performance.

    Parameters
    ----------
    G : torch.Tensor
        Input matrix with shape (M, N) or batched input with shape (B, M, N).

    Returns
    -------
    torch.Tensor
        Orthogonalized tensor in bfloat16 with same shape as input.

    Raises
    ------
    ValueError
        If input is not 2D or 3D.
    """
    if G.ndim == 2:
        return _zeropower_via_newtonschulz5_2d(G)
    if G.ndim == 3:
        return _zeropower_via_newtonschulz5_3d(G)
    raise ValueError("Input must be 2D or 3D for Newton-Schulz orthogonalization.")


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
        Default is 10.0 (Adam lr = lr/10).
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
        lr_adjust: float = 10.0,
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
        # Static parameter routing: built once on first step() call.
        self._routing_built = False
        self._routing: list[dict[str, Any]] = []

    def _build_param_routing(self) -> None:
        """
        Classify parameters into Muon and Adam routes (static routing).

        Routing logic:
        - >=2D parameters → Muon path (Newton-Schulz + momentum)
        - 1D parameters → Adam path (standard Adam update)
        """
        if self._routing_built:
            return

        self._routing = []
        for group in self.param_groups:
            muon_params: list[dict[str, Any]] = []
            adam_params: list[dict[str, Any]] = []

            for p in group["params"]:
                if p.ndim >= 2:
                    muon_params.append(
                        {
                            "param": p,
                            "rows": int(p.shape[0]),
                            "cols": int(p.numel() // p.shape[0]),
                        }
                    )
                else:
                    adam_params.append({"param": p})

            self._routing.append(
                {
                    "muon_params": muon_params,
                    "adam_params": adam_params,
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
            adam_params: list[torch.Tensor] = []
            adam_grads_fp32: list[torch.Tensor] = []
            adam_exp_avgs: list[torch.Tensor] = []
            adam_exp_avg_sqs: list[torch.Tensor] = []
            adam_states: list[dict[str, Any]] = []

            for entry in route["adam_params"]:
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
                adam_lr = lr if lr_adjust <= 0 else lr / lr_adjust

                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                torch._foreach_lerp_(adam_exp_avgs, adam_grads_fp32, 1 - adam_betas[0])
                grad_sq = torch._foreach_mul(adam_grads_fp32, adam_grads_fp32)
                torch._foreach_lerp_(adam_exp_avg_sqs, grad_sq, 1 - adam_betas[1])

                for i, p in enumerate(adam_params):
                    state = adam_states[i]
                    bias_corr1 = 1 - state["beta1_pow"]
                    bias_corr2 = 1 - state["beta2_pow"]
                    step_size = adam_lr / bias_corr1
                    # delta = -step_size * m_hat / (sqrt(v_hat) + eps)
                    denom = (adam_exp_avg_sqs[i] / bias_corr2).sqrt().add_(ADAM_EPS)
                    delta_fp32 = -step_size * (adam_exp_avgs[i] / denom)
                    p.add_(delta_fp32.to(p.dtype))

            # === Step 2. Muon update for >=2D parameters (weight matrices) ===
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

            if weight_decay > 0 and muon_params_for_decay:
                torch._foreach_mul_(muon_params_for_decay, 1.0 - lr * weight_decay)

            if not active_entries:
                continue

            # m_t = beta * m_{t-1} + (1 - beta) * g_t
            torch._foreach_lerp_(muon_momentum_buffers, muon_grads, 1 - momentum)
            # update = beta * m_t + (1 - beta) * g_t
            muon_updates = torch._foreach_lerp(
                muon_grads, muon_momentum_buffers, momentum
            )

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

            for (rows, cols, _device, dtype), bucket_entries in buckets.items():
                # scale = coeff * sqrt(max(m, n))  [match-RMS mode]
                # scale = sqrt(max(1, m/n))        [rectangular mode]
                if lr_adjust <= 0:
                    scale = lr_adjust_coeff * math.sqrt(float(max(rows, cols)))
                else:
                    scale = max(1.0, rows / cols) ** 0.5

                if len(bucket_entries) == 1:
                    entry, update_tensor = bucket_entries[0]
                    update_matrix = update_tensor.reshape(rows, cols)
                    if not update_matrix.is_contiguous():
                        update_matrix = update_matrix.contiguous()

                    orth = _zeropower_via_newtonschulz5_2d(update_matrix)
                    orth.mul_(scale)
                    delta = orth.reshape(entry["param"].shape)
                    if delta.dtype != dtype:
                        delta = delta.to(dtype)
                    entry["param"].add_(delta, alpha=-lr)
                    continue

                matrices: list[torch.Tensor] = []
                params: list[torch.Tensor] = []
                orig_shapes: list[tuple[int, ...]] = []

                for entry, update_tensor in bucket_entries:
                    update_matrix = update_tensor.reshape(rows, cols)
                    matrices.append(
                        update_matrix
                        if update_matrix.is_contiguous()
                        else update_matrix.contiguous()
                    )
                    params.append(entry["param"])
                    orig_shapes.append(entry["param"].shape)

                stacked = torch.stack(matrices, dim=0)
                orth = _zeropower_via_newtonschulz5_3d(stacked)
                orth.mul_(scale)
                if orth.dtype != dtype:
                    orth = orth.to(dtype)

                for i, _ in enumerate(bucket_entries):
                    params[i].add_(orth[i].reshape(orig_shapes[i]), alpha=-lr)

        return loss
