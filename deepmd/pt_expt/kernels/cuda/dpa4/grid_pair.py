# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused DPA4 / SeZM grid pair product.

The CUDA operator ``deepmd::dpa4_grid_pair`` (see
``source/op/pt/dpa4_grid_pair.cu``) evaluates
``from_grid(to_grid(left) * to_grid(right))`` without materializing the grid
field. That expression is the core of every grid operator of the model: the
parameter-free node product, the polynomial grid MLP, and the branch mixer at a
single branch, where its softmax router is identically one.

At the production SO(3) shape the grid field is 39 times larger than the
coefficient operand that produces it, so the unfused form is dominated by
writing and rereading it -- plus, because the projection is expressed as an
einsum over non-adjacent axes, by full-size layout copies around each multiply.
"""

from __future__ import (
    annotations,
)

from functools import (
    cache,
)
from typing import (
    Any,
)

import torch

__all__ = [
    "SUPPORTED_SLOTS",
    "ensure_registered",
    "grid_pair",
    "op_available",
]

# Coefficient-slot counts the operator is instantiated for, mirroring
# ``DPA4_GRID_FOR_EACH_P`` in ``source/op/pt/dpa4/grid_pair.cu``. ``P`` is the
# coefficient dimension times the frame count, so the SO(3) grids of degrees one
# to six give ``3 * (l + 1)^2`` and 9 is the matching S2 grid.
SUPPORTED_SLOTS = frozenset({9, 12, 27, 48, 75, 108, 147})


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa4_grid_pair`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa4_grid_pair", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def _forward_fake(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> torch.Tensor:
    del right, to_grid, from_grid
    return torch.empty_like(left)


def _backward_fake(
    grad_out: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_out, to_grid, from_grid
    return torch.empty_like(left), torch.empty_like(right)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _backward(ctx: Any, grad_out: torch.Tensor) -> tuple:
    left, right, to_grid, from_grid = ctx.saved_tensors
    g_left, g_right = torch.ops.deepmd.dpa4_grid_pair_backward(
        grad_out.contiguous(), left, right, to_grid, from_grid
    )
    return g_left, g_right, None, None


@cache
def _register_ops() -> None:
    """Register fake and autograd implementations once."""
    torch.library.register_fake("deepmd::dpa4_grid_pair")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4_grid_pair_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4_grid_pair", _backward, setup_context=_setup_context
    )


def ensure_registered() -> None:
    """Register fake and autograd implementations when the op is available."""
    if op_available():
        _register_ops()


def grid_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate ``from_grid(to_grid(left) * to_grid(right))`` on coefficients.

    Parameters
    ----------
    left, right : torch.Tensor
        Coefficient operands with shape (N, P, C), where ``P`` is the product of
        the coefficient dimension and the frame count.
    to_grid : torch.Tensor
        Coefficient-to-grid projector with shape (G, P).
    from_grid : torch.Tensor
        Grid-to-coefficient projector, transposed to shape (G, P).

    Returns
    -------
    torch.Tensor
        Coefficient result with shape (N, P, C).
    """
    ensure_registered()
    return torch.ops.deepmd.dpa4_grid_pair(left, right, to_grid, from_grid)
