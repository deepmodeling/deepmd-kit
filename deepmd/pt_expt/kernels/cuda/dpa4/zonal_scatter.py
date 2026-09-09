# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused DPA4 / SeZM geometric initial embedding.

The CUDA operator ``deepmd::dpa4_zonal_scatter`` (see
``source/op/pt/dpa4/zonal_scatter.cu``) evaluates the per-edge message of the
initial embedding and its destination reduction in one pass::

    out[n, r, c] = sum_{dst[e] = n} zonal[e, r] * radial[e, slot[r], c]

The reference composition writes the message as an ``(E, R, C)`` tensor before
scattering it, which is 1.3 GB at the production shape and dominates the cost of
a step that is otherwise two small operand reads. The fused form keeps the
message in registers and walks the destination CSR the convolution already
builds, so the reduction is also atomic free and bitwise reproducible.
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
    "ensure_registered",
    "op_available",
    "zonal_scatter",
]

# Degrees with an instantiation, mirroring ``DPA4_ZONAL_FOR_EACH_LMAX`` in
# ``source/op/pt/dpa4/zonal_scatter.cu``.
_MAX_LMAX = 6


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa4_zonal_scatter`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa4_zonal_scatter", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def _forward_fake(
    zonal: torch.Tensor,
    radial: torch.Tensor,
    dst: torch.Tensor,
    dst_order: torch.Tensor,
    dst_rowptr: torch.Tensor,
    node_scale: torch.Tensor,
    node_count: int | torch.SymInt,
) -> torch.Tensor:
    del dst, dst_order, dst_rowptr, node_scale
    return zonal.new_empty((node_count, zonal.shape[1] + 1, radial.shape[2]))


def _backward_fake(
    grad_out: torch.Tensor,
    zonal: torch.Tensor,
    radial: torch.Tensor,
    dst: torch.Tensor,
    node_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_out, dst, node_scale
    return torch.empty_like(zonal), torch.empty_like(radial)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    zonal, radial, dst = inputs[:3]
    # The output is saved so the backward can recover the unscaled reduction,
    # which is what the scale's own cotangent contracts against.
    ctx.save_for_backward(zonal, radial, dst, inputs[5], output)


def _backward(ctx: Any, grad_out: torch.Tensor) -> tuple:
    zonal, radial, dst, node_scale, out = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    g_zonal, g_radial = torch.ops.deepmd.dpa4_zonal_scatter_backward(
        grad_out, zonal, radial, dst, node_scale
    )
    # ``out = scale * acc``, and the degree floor keeps ``scale`` strictly
    # positive, so the unscaled reduction is recovered exactly by division.
    g_scale = (grad_out * out).sum(dim=(1, 2)) / node_scale.reshape(-1)
    return g_zonal, g_radial, None, None, None, g_scale, None


@cache
def _register_ops() -> None:
    """Register fake and autograd implementations once."""
    torch.library.register_fake("deepmd::dpa4_zonal_scatter")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4_zonal_scatter_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4_zonal_scatter", _backward, setup_context=_setup_context
    )


def ensure_registered() -> None:
    """Register fake and autograd implementations when the op is available."""
    if op_available():
        _register_ops()


def zonal_scatter(
    zonal: torch.Tensor,
    radial: torch.Tensor,
    dst: torch.Tensor,
    dst_order: torch.Tensor,
    dst_rowptr: torch.Tensor,
    node_scale: torch.Tensor,
    node_count: int | torch.SymInt,
) -> torch.Tensor:
    """
    Reduce the geometric initial message onto its destination nodes.

    Parameters
    ----------
    zonal : torch.Tensor
        Zonal coupling with shape (E, R), ``R = (lmax + 1) ** 2 - 1`` packed
        non-scalar rows of degrees 1 to ``lmax``.
    radial : torch.Tensor
        Per-edge radial features with shape (E, L, C), ``L > lmax``. Row ``r``
        of the message reuses the radial feature of its own degree.
    dst : torch.Tensor
        Destination node index of every edge with shape (E,).
    dst_order : torch.Tensor
        Stable sorting permutation of ``dst`` with shape (E,).
    dst_rowptr : torch.Tensor
        Destination row pointer with shape (node_count + 1,).
    node_scale : torch.Tensor
        Smooth degree normalization with shape (node_count,), applied on the way
        out. It descends from the cutoff envelope and is differentiated.
    node_count : int or torch.SymInt
        Number of destination nodes.

    Returns
    -------
    torch.Tensor
        Node aggregate in the packed layout, normalized, with shape
        ``(node_count, R + 1, C)``. Row zero is the scalar coefficient, which
        this embedding leaves at zero.
    """
    ensure_registered()
    return torch.ops.deepmd.dpa4_zonal_scatter(
        zonal, radial, dst, dst_order, dst_rowptr, node_scale, node_count
    )


def supported(lmax: int, n_row: int, channels: int) -> bool:
    """Whether the operator is instantiated for this embedding shape."""
    return 1 <= lmax <= _MAX_LMAX and n_row == (lmax + 1) ** 2 - 1 and channels > 0
