# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Sequence,
)

import torch


def _nonuniform_like(tensor: torch.Tensor) -> torch.Tensor:
    """Create deterministic nonuniform weights for gradient projections."""
    return torch.linspace(
        -0.5,
        0.75,
        tensor.numel(),
        dtype=tensor.dtype,
        device=tensor.device,
    ).reshape_as(tensor)


def _project_first_grads(
    descriptor_tensor: torch.Tensor,
    grad_inputs: Sequence[torch.Tensor],
    dy_tensor: torch.Tensor,
    weights: Sequence[torch.Tensor],
) -> torch.Tensor:
    first_grads = torch.autograd.grad(
        descriptor_tensor,
        grad_inputs,
        grad_outputs=dy_tensor,
        retain_graph=True,
    )
    projection = descriptor_tensor.new_zeros(())
    for grad, weight in zip(first_grads, weights, strict=True):
        projection = projection + (grad * weight).sum()
    return projection


def assert_second_order_backward_matches_finite_difference(
    descriptor_tensor: torch.Tensor,
    grad_inputs: torch.Tensor | Sequence[torch.Tensor],
) -> None:
    """Compare dz/dy from the grad-grad kernel with a finite-difference reference."""
    if isinstance(grad_inputs, torch.Tensor):
        grad_inputs = (grad_inputs,)

    dy_tensor = _nonuniform_like(descriptor_tensor).requires_grad_(True)
    first_grads = torch.autograd.grad(
        descriptor_tensor,
        grad_inputs,
        grad_outputs=dy_tensor,
        create_graph=True,
        retain_graph=True,
    )
    weights = tuple(_nonuniform_like(grad) for grad in first_grads)
    projection = descriptor_tensor.new_zeros(())
    for grad, weight in zip(first_grads, weights, strict=True):
        projection = projection + (grad * weight).sum()

    dz_dy_tensor = torch.autograd.grad(
        projection,
        dy_tensor,
        retain_graph=True,
    )[0]

    eps, atol, rtol = (
        (1e-6, 1e-6, 1e-6)
        if descriptor_tensor.dtype == torch.float64
        else (1e-2, 5e-3, 5e-3)
    )
    reference = torch.empty_like(dy_tensor)
    base_dy = dy_tensor.detach()
    reference_flat = reference.view(-1)
    for idx in range(dy_tensor.numel()):
        dy_plus = base_dy.clone()
        dy_minus = base_dy.clone()
        dy_plus.view(-1)[idx] += eps
        dy_minus.view(-1)[idx] -= eps

        plus_projection = _project_first_grads(
            descriptor_tensor, grad_inputs, dy_plus, weights
        )
        minus_projection = _project_first_grads(
            descriptor_tensor, grad_inputs, dy_minus, weights
        )
        reference_flat[idx] = (plus_projection - minus_projection) / (2.0 * eps)

    torch.testing.assert_close(
        dz_dy_tensor,
        reference,
        atol=atol,
        rtol=rtol,
    )
