# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""Force and per-atom virial assembly from the per-edge energy gradient.

The force on an extended atom is the sum of the energy gradient over the edges
that end on it minus the sum over the edges that start on it, and the per-atom
virial is the corresponding sum of ``-0.5 * g (x) v`` outer products. Both are
expressed as two segmented reductions over pre-built CSR topologies, one per
endpoint, rather than as four scatters: the segmented form is contention free
and, because each node's contributions are summed in one block, the summation
order is fixed.

Accumulation is in float64. The outer product is recomputed per edge from the
three-component gradient and displacement and is never materialized, which
removes an ``(E, 9)`` intermediate.

Operator boundary
-----------------
The kernel is exposed as a functional ``custom_op`` paired with an explicit
closed-form backward operator, so it survives the ``make_fx`` force-autograd
trace and can be replayed under :func:`torch.no_grad` when the frozen inference
graph runs. A closed form -- rather than a nested :func:`torch.autograd.grad` --
is required because the backward operator is dispatched below autograd during
that replay. A ``custom_op`` is opaque to Inductor: nothing inside it fuses with
the surrounding graph and its buffers are invisible to the memory planner, so
only tensors that must cross the boundary do.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..common import CUTILE_AVAILABLE
from .tile_configs import tile_config

if CUTILE_AVAILABLE:
    import cuda.tile as ct

__all__ = ["edge_force_assembly"]

#: Extended atoms owned by one block. Widening the block amortizes the segment
#: bound read, which dominates when a block owns a single short segment; the
#: per-atom loop is serial, so widening too far loses again. Two atoms per block
#: measured 2.4x faster than one on the production distribution.
NODES_PER_BLOCK = 2


if CUTILE_AVAILABLE:

    @ct.kernel
    def _force_segment(
        grad,
        edge_vec,
        order,
        row_ptr,
        force,
        virial,
        sign: ct.Constant[float],
        accumulate: ct.Constant[int],
        BE: ct.Constant[int],
        NODES: ct.Constant[int],
    ):
        """Reduce one endpoint's contribution to a run of extended atoms.

        A block owns ``NODES`` consecutive atoms rather than one. Both endpoints
        range over the *extended* atoms, so the mean segment holds only a handful
        of edges: with one atom per block the two dependent scalar reads of the
        segment bounds are longer than the work they gate, and the kernel runs at
        a fixed cost per block rather than at its bandwidth. Widening the block
        amortizes that latency and turns the bounds into a single coalesced read.

        The three force components and the nine virial components are carried as
        four- and sixteen-lane tiles so both stay vectorized on power-of-two
        extents; the unused lanes are loaded as zeros and their stores fall
        outside the output rows, where they are discarded. The same applies to
        the atoms of a trailing partial block: their bounds gather out of range
        as zeros, which yields an empty segment.
        """
        base = ct.bid(0) * NODES
        # Two gathers rather than one window of ``NODES + 1``: a tile extent must
        # be a power of two, and both of these are coalesced reads of the same
        # cache lines.
        lane = ct.arange(NODES, dtype=ct.int32)
        starts = ct.gather(row_ptr, base + lane)
        stops = ct.gather(row_ptr, base + 1 + lane)
        for index in range(NODES):
            start = ct.extract(starts, (index,), (1,)).item()
            stop = ct.extract(stops, (index,), (1,)).item()
            acc_force = ct.zeros((4,), dtype=ct.float64)
            acc_virial = ct.zeros((4, 4), dtype=ct.float64)
            for position in range(start, stop, BE):
                slot = position + ct.arange(BE, dtype=ct.int32)
                live = slot < stop
                entry = ct.gather(
                    order, ct.where(live, slot, stop - 1), check_bounds=False
                )
                keep = ct.where(live.reshape((BE, 1)), 1.0, 0.0)
                g = (
                    ct.load_advanced_indexing(
                        grad, (entry, ct.Slice(0, 4)), padding_mode=ct.PaddingMode.ZERO
                    )
                    * keep
                )
                v = (
                    ct.load_advanced_indexing(
                        edge_vec,
                        (entry, ct.Slice(0, 4)),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    * keep
                )
                acc_force = acc_force + ct.sum(g.astype(ct.float64), axis=0)
                outer = g.reshape((BE, 4, 1)) * v.reshape((BE, 1, 4))
                acc_virial = acc_virial - 0.5 * ct.sum(outer.astype(ct.float64), axis=0)
            node = base + index
            out_force = (acc_force * sign).astype(ct.float32)
            out_virial = acc_virial.astype(ct.float32).reshape((1, 16))
            if accumulate:
                out_force = out_force + ct.reshape(
                    ct.load(force, (node, 0), (1, 4), padding_mode=ct.PaddingMode.ZERO),
                    (4,),
                )
                out_virial = out_virial + ct.load(
                    virial, (node, 0), (1, 16), padding_mode=ct.PaddingMode.ZERO
                )
            ct.store(force, (node, 0), ct.reshape(out_force, (1, 4)))
            ct.store(virial, (node, 0), out_virial)


def _launch_forward(
    grad: Tensor,
    edge_vec: Tensor,
    dst_order: Tensor,
    dst_row_ptr: Tensor,
    src_order: Tensor,
    src_row_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Assemble the force and per-atom virial from the per-edge energy gradient.

    Parameters
    ----------
    grad : Tensor
        Per-edge energy gradient with respect to the displacement, ``(E, 3)``.
    edge_vec : Tensor
        Per-edge displacement, ``(E, 3)``.
    dst_order, dst_row_ptr : Tensor
        Destination-sorted edge order and its CSR offsets over extended atoms.
    src_order, src_row_ptr : Tensor
        Source-sorted edge order and its CSR offsets over extended atoms.

    Returns
    -------
    tuple[Tensor, Tensor]
        Force ``(N_ext, 3)`` and per-atom virial ``(N_ext, 9)``.

    Notes
    -----
    The two endpoint passes run as separate launches over the same output, the
    second accumulating. Running them as one kernel would need both topologies
    resident and buys nothing: each pass is a streaming reduction.
    """
    config = tile_config("force_assembly")
    n_ext = dst_row_ptr.shape[0] - 1
    # The four- and sixteen-lane tiles the kernel uses address one padded column
    # past the physical layout, so the buffers carry that padding and are sliced
    # on return.
    grad_pad = grad.new_zeros((grad.shape[0], 4))
    grad_pad[:, :3] = grad
    vec_pad = edge_vec.new_zeros((edge_vec.shape[0], 4))
    vec_pad[:, :3] = edge_vec
    force = grad.new_empty((n_ext, 4))
    virial = grad.new_empty((n_ext, 16))
    stream = torch.cuda.current_stream()
    for order, row_ptr, sign, accumulate in (
        (dst_order, dst_row_ptr, 1.0, 0),
        (src_order, src_row_ptr, -1.0, 1),
    ):
        ct.launch(
            stream,
            (math.ceil(n_ext / NODES_PER_BLOCK),),
            _force_segment,
            (
                grad_pad,
                vec_pad,
                order.to(torch.int32),
                row_ptr.to(torch.int32),
                force,
                virial,
                sign,
                accumulate,
                config.tile,
                NODES_PER_BLOCK,
            ),
        )
    return (
        force[:, :3].contiguous(),
        virial.reshape(n_ext, 4, 4)[:, :3, :3].reshape(n_ext, 9).contiguous(),
    )


@torch.library.custom_op("sezm_cutile::edge_force_assembly", mutates_args=())
def _force_op(
    grad: Tensor,
    edge_vec: Tensor,
    dst_order: Tensor,
    dst_row_ptr: Tensor,
    src_order: Tensor,
    src_row_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    return _launch_forward(
        grad.contiguous(),
        edge_vec.contiguous(),
        dst_order,
        dst_row_ptr,
        src_order,
        src_row_ptr,
    )


@_force_op.register_fake
def _(grad, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr):
    n_ext = dst_row_ptr.shape[0] - 1
    return grad.new_empty((n_ext, 3)), grad.new_empty((n_ext, 9))


def edge_force_assembly(
    grad: Tensor,
    edge_vec: Tensor,
    dst_order: Tensor,
    dst_row_ptr: Tensor,
    src_order: Tensor,
    src_row_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Assemble the force and per-atom virial from the per-edge energy gradient."""
    return _force_op(grad, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr)
