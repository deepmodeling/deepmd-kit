# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""Fused cuTile attention aggregation for the SO(2) message.

The forward folds four stages into one destination-segmented pass: the
block-diagonal inverse rotation back to the global frame, the inverse-rotation
degree rescale, the per-edge envelope-gated softmax weight, and the reduction
onto the destination node. Neither the rotated-back message nor the weighted
value is written to DRAM.

The forward grid is one block per destination node walking that node's CSR
segment, which is deterministic and avoids the atomic scatter that would
serialize on the order of a hundred colliding edges per atom. The backward is
edge major -- every gradient it produces is per edge -- so it needs no topology
at all.

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

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
)

import torch
from torch import (
    Tensor,
)

from ..common import (
    CUTILE_AVAILABLE,
    Emitter,
    generated_module,
    kernel_variant,
)
from .indexing import (
    SO2TileLayout,
    m_major_index,
    rotation_pairs,
)
from .tile_configs import (
    tile_config,
)

if TYPE_CHECKING:
    from types import (
        ModuleType,
    )

if CUTILE_AVAILABLE:
    import cuda.tile as ct

__all__ = ["build_row_ptr", "flash_atten_aggregate"]


def build_row_ptr(sorted_key: Tensor, n_nodes: int) -> Tensor:
    """Build CSR row offsets ``(N + 1,)`` from an ascending segment key.

    Parameters
    ----------
    sorted_key : Tensor
        Segment key of each edge in ascending order, ``(E,)``.
    n_nodes
        Number of segments. May be a ``SymInt``, which keeps the node axis
        unspecialized under ``make_fx``.

    Returns
    -------
    Tensor
        Segment offsets in int32.

    Notes
    -----
    ``searchsorted`` on the sorted key is the traceable, allocation-light way to
    obtain segment boundaries: it lowers cleanly under ``make_fx`` and needs no
    data-dependent control flow. The offsets are int32 because a ``range`` whose
    bounds and step disagree in width fails tile-compiler verification, and every
    kernel here derives its segment loop bounds from these offsets.
    """
    boundaries = torch.arange(
        n_nodes + 1, device=sorted_key.device, dtype=sorted_key.dtype
    )
    return torch.searchsorted(sorted_key, boundaries).to(torch.int32)


_HEADER = '''# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generated cuTile attention aggregation: lmax={lmax} Cf={cf} F={focus} H={heads}."""

from typing import Annotated

import cuda.tile as ct

BigArray = Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64)]
BE = {be}
BS = {bs}
CF = {cf}
CW = {cw}
ROW = {row}
'''


def _generate(
    layout: SO2TileLayout,
    n_focus: int,
    n_head: int,
    block_edges: int,
    segment_edges: int,
    rescale: list[float],
) -> str:
    """Return the source of the aggregation forward and backward kernels."""
    cf = layout.focus_dim
    c_wide = n_focus * cf
    head_dim = cf // n_head
    n_row = layout.n_m0 + layout.n_m1
    pairs = rotation_pairs(layout.lmax)
    coeff = m_major_index(layout.lmax)
    by_full: dict[int, list[tuple[int, int]]] = {}
    by_row: dict[int, list[tuple[int, int]]] = {}
    for slot, (reduced, full) in enumerate(pairs):
        by_full.setdefault(full, []).append((slot, reduced))
        by_row.setdefault(reduced, []).append((slot, full))
    full_rows = sorted(by_full)

    source = [
        _HEADER.format(
            lmax=layout.lmax,
            cf=cf,
            focus=n_focus,
            heads=n_head,
            be=block_edges,
            bs=segment_edges,
            cw=c_wide,
            row=layout.row,
        )
    ]

    def emit_coefficients(emit: Emitter, index: str) -> None:
        """Load the transposed rotation coefficients of one edge tile."""
        for slot, (reduced, full) in enumerate(pairs):
            emit(
                f"d{slot} = ct.reshape(ct.load_advanced_indexing(wigner_t,"
                f" ({index}, ct.Slice({full * layout.dim + coeff[reduced]}, 1)),"
                " padding_mode=ct.PaddingMode.ZERO), (BE, 1))"
            )

    def head_weight(focus: int) -> str:
        """Return the per-edge attention weight of one focus stream.

        The weight is one scalar per attention head, broadcast over that head's
        channels; with a single head the broadcast is over the whole stream.
        """
        if n_head == 1:
            return (
                f"ct.reshape(ct.load_advanced_indexing(alpha, (entry,"
                f" ct.Slice({focus * n_head}, 1)),"
                " padding_mode=ct.PaddingMode.ZERO), (BE, 1))"
            )
        parts = [
            f"ct.reshape(ct.load_advanced_indexing(alpha, (entry,"
            f" ct.Slice({focus * n_head + head}, 1)),"
            " padding_mode=ct.PaddingMode.ZERO), (BE, 1))"
            for head in range(n_head)
        ]
        return " + ".join(f"{part} * head{head}" for head, part in enumerate(parts))

    # === Forward: one block per destination node ===
    emit = Emitter()
    emit("node = ct.bid(0)")
    emit("start = ct.load(row_ptr, (node,), (1,)).item()")
    emit("stop = ct.load(row_ptr, (node + 1,), (1,)).item()")
    if n_head > 1:
        for head in range(n_head):
            lo, hi = head * head_dim, (head + 1) * head_dim
            emit(
                f"head{head} = ct.where("
                f"(ct.arange(CF, dtype=ct.int32) >= {lo})"
                f" & (ct.arange(CF, dtype=ct.int32) < {hi}), 1.0, 0.0"
                ").reshape((1, CF))"
            )
    for focus in range(n_focus):
        for full in full_rows:
            emit(f"acc{focus}_{full} = ct.zeros((CF,), dtype=ct.float32)")
    emit("for position in range(start, stop, BS):")
    inner = Emitter(indent="        ")
    inner("slot = position + ct.arange(BS, dtype=ct.int32)")
    inner("live = slot < stop")
    inner(
        "entry = ct.gather(order, ct.where(live, slot, stop - 1), check_bounds=False)"
    )
    emit_coefficients(inner, "entry")
    for focus in range(n_focus):
        inner(f"weight = ct.where(live.reshape((BS, 1)), {head_weight(focus)}, 0.0)")
        for reduced in range(n_row):
            inner(
                f"v{reduced} = ct.load_advanced_indexing(xlocal,"
                f" (entry, ct.Slice({focus * n_row * cf + reduced * cf}, CF)),"
                " padding_mode=ct.PaddingMode.ZERO)"
            )
        for full in full_rows:
            terms = " + ".join(
                f"d{slot} * v{reduced}" for slot, reduced in by_full[full]
            )
            inner(
                f"acc{focus}_{full} = acc{focus}_{full} + ct.sum(({terms}) * weight,"
                f" axis=0) * {rescale[full]!r}"
            )
    emit.extend([line[4:] for line in inner.lines])
    for focus in range(n_focus):
        for full in range(layout.dim):
            if full in by_full:
                emit(
                    f"ct.store(out, (node, {full}, {focus}, 0),"
                    f" ct.reshape(acc{focus}_{full}, (1, 1, 1, CF)))"
                )
            else:
                emit(
                    f"ct.store(out, (node, {full}, {focus}, 0),"
                    " ct.reshape(ct.zeros((CF,), dtype=ct.float32), (1, 1, 1, CF)))"
                )
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def flash_forward(xlocal: BigArray, wigner_t: BigArray, alpha,",
                "                  order, row_ptr, out: BigArray):",
                '    """Rotate back, weight and reduce one destination node."""',
            ],
        )
    )

    # === Backward: one block per edge tile ===
    emit = Emitter()
    emit("edge = ct.bid(0)")
    emit("entry = edge * BE + ct.arange(BE, dtype=ct.int32)")
    emit("dstn = ct.load(dsts, (edge,), (BE,), padding_mode=ct.PaddingMode.ZERO)")
    if n_head > 1:
        for head in range(n_head):
            lo, hi = head * head_dim, (head + 1) * head_dim
            emit(
                f"head{head} = ct.where("
                f"(ct.arange(CF, dtype=ct.int32) >= {lo})"
                f" & (ct.arange(CF, dtype=ct.int32) < {hi}), 1.0, 0.0"
                ").reshape((1, CF))"
            )
    emit_coefficients(emit, "entry")
    for focus in range(n_focus):
        emit("")
        emit(f"# === Focus stream {focus} ===")
        emit(f"weight = {head_weight(focus)}")
        for full in full_rows:
            emit(
                f"g{full} = ct.load_advanced_indexing(gout,"
                f" (dstn, ct.Slice({(full * n_focus + focus) * cf}, CF)),"
                f" padding_mode=ct.PaddingMode.ZERO) * {rescale[full]!r}"
            )
        for reduced in range(n_row):
            emit(
                f"v{reduced} = ct.load_advanced_indexing(xlocal,"
                f" (entry, ct.Slice({focus * n_row * cf + reduced * cf}, CF)),"
                " padding_mode=ct.PaddingMode.ZERO)"
            )
        # The message is linear in the local feature, in the rotation and in the
        # weight, so each gradient is the product of the other two contracted
        # over the axes it does not carry.
        for reduced in range(n_row):
            terms = " + ".join(f"d{slot} * g{full}" for slot, full in by_row[reduced])
            emit(
                f"ct.store_advanced_indexing(gxlocal,"
                f" (entry, ct.Slice({focus * n_row * cf + reduced * cf}, CF)),"
                f" weight * ({terms}))"
            )
        for slot, (reduced, full) in enumerate(pairs):
            emit(f"gd{slot}_{focus} = ct.sum(weight * g{full} * v{reduced}, axis=1)")
        for head in range(n_head):
            # The weight gradient contracts the rotated-back message against the
            # output gradient over the channels this head owns.
            span = "" if n_head == 1 else f" * head{head}"
            terms = " + ".join(
                "ct.sum(g{} * ({}){}, axis=1)".format(
                    full,
                    " + ".join(
                        f"d{slot} * v{reduced}" for slot, reduced in by_full[full]
                    ),
                    span,
                )
                for full in full_rows
            )
            emit(
                f"ct.store(galpha, (edge, {focus}, {head}),"
                f" ct.reshape({terms}, (BE, 1, 1)))"
            )
    emit("")
    emit("# === Rotation gradient, summed over the focus streams that share it ===")
    for slot, (reduced, full) in enumerate(pairs):
        total = " + ".join(f"gd{slot}_{focus}" for focus in range(n_focus))
        emit(
            f"ct.store(gwigner_t, (edge, {full}, {coeff[reduced]}),"
            f" ct.reshape({total}, (BE, 1, 1)))"
        )
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def flash_backward(xlocal: BigArray, wigner_t: BigArray, alpha,",
                "                   dsts, gout: BigArray, gxlocal: BigArray,",
                "                   gwigner_t: BigArray, galpha):",
                '    """Emit the local-feature, rotation and attention-weight gradients."""',
            ],
        )
    )
    return "".join(source)


def _module(
    layout: SO2TileLayout,
    n_focus: int,
    n_head: int,
    rescale: tuple[float, ...],
) -> ModuleType:
    """Return the generated module for the resolved forward and backward tiles.

    Both tiles size the same module, so a change to either regenerates it; the
    source digest keeps the two variants distinct in the cache.
    """
    block_edges = tile_config("flash_bwd", layout.key).tile
    segment_edges = tile_config("flash_fwd", layout.key).tile
    stem = (
        f"sezm_flash_l{layout.lmax}_c{layout.focus_dim}"
        f"_f{n_focus}_h{n_head}_b{block_edges}_s{segment_edges}"
    )
    return generated_module(
        stem,
        _generate(layout, n_focus, n_head, block_edges, segment_edges, list(rescale)),
    )


def _launch_forward(
    x_local: Tensor,
    wigner_t: Tensor,
    rescale: tuple[float, ...],
    alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    layout: SO2TileLayout,
    n_focus: int,
    n_head: int,
) -> Tensor:
    """Rotate back, weight by the attention softmax and reduce onto destinations.

    Parameters
    ----------
    x_local : Tensor
        Per-focus local features, ``(E, F, D_m, Cf)``.
    wigner_t : Tensor
        Transposed block-diagonal Wigner-D per edge, ``(E, D, D)``.
    rescale : tuple[float, ...]
        Inverse-rotation degree rescale, one entry per full-basis row. Baked into
        the kernel, so a configuration change regenerates it.
    alpha : Tensor
        Envelope-gated softmax weight, ``(E, F, H)``.
    order : Tensor
        Edge indices sorted by destination, ``(E,)``.
    row_ptr : Tensor
        Destination CSR offsets, ``(N + 1,)``.
    layout : SO2TileLayout
        Configuration geometry.
    n_focus, n_head : int
        Focus stream and attention head counts.

    Returns
    -------
    Tensor
        Ungated aggregate ``(N, D, C_wide)``.
    """
    n_node = row_ptr.shape[0] - 1
    out = x_local.new_empty((n_node, layout.dim, n_focus, layout.focus_dim))
    config = tile_config("flash_fwd", layout.key)
    module = _module(layout, n_focus, n_head, rescale)
    ct.launch(
        torch.cuda.current_stream(),
        (n_node,),
        kernel_variant(module.flash_forward, **config.hints),
        (
            x_local.reshape(x_local.shape[0], -1),
            wigner_t.reshape(wigner_t.shape[0], -1),
            alpha.reshape(alpha.shape[0], -1),
            order.to(torch.int32),
            row_ptr.to(torch.int32),
            out,
        ),
    )
    return out.reshape(n_node, layout.dim, n_focus * layout.focus_dim)


def _launch_backward(
    grad_out: Tensor,
    x_local: Tensor,
    wigner_t: Tensor,
    rescale: tuple[float, ...],
    alpha: Tensor,
    dst: Tensor,
    layout: SO2TileLayout,
    n_focus: int,
    n_head: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the local-feature, rotation and attention-weight gradients."""
    n_edge = x_local.shape[0]
    grad_local = torch.empty_like(x_local)
    grad_wigner = torch.zeros_like(wigner_t)
    grad_alpha = torch.empty_like(alpha)
    config = tile_config("flash_bwd", layout.key)
    module = _module(layout, n_focus, n_head, rescale)
    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(n_edge / config.tile),),
        kernel_variant(module.flash_backward, **config.hints),
        (
            x_local.reshape(n_edge, -1),
            wigner_t.reshape(n_edge, -1),
            alpha.reshape(n_edge, -1),
            dst.to(torch.int32),
            grad_out.reshape(grad_out.shape[0], -1),
            grad_local.reshape(n_edge, -1),
            grad_wigner,
            grad_alpha,
        ),
    )
    return grad_local, grad_wigner, grad_alpha


@torch.library.custom_op("sezm_cutile::flash_atten_aggregate", mutates_args=())
def _flash_op(
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> Tensor:
    n_focus, focus_dim = x_local.shape[1], x_local.shape[3]
    layout = SO2TileLayout(lmax=lmax, focus_dim=focus_dim, n_layers=2)
    return _launch_forward(
        x_local.contiguous(),
        wigner_dt,
        tuple(rescale.tolist()),
        alpha.contiguous(),
        order.contiguous(),
        row_ptr.contiguous(),
        layout,
        n_focus,
        n_head,
    )


@_flash_op.register_fake
def _(x_local, wigner_dt, rescale, alpha, order, row_ptr, dst, lmax, n_head):
    n_focus, focus_dim = x_local.shape[1], x_local.shape[3]
    return x_local.new_empty(
        (row_ptr.shape[0] - 1, (lmax + 1) ** 2, n_focus * focus_dim)
    )


@torch.library.custom_op("sezm_cutile::flash_atten_aggregate_bwd", mutates_args=())
def _flash_bwd_op(
    grad_out: Tensor,
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> tuple[Tensor, Tensor, Tensor]:
    n_focus, focus_dim = x_local.shape[1], x_local.shape[3]
    layout = SO2TileLayout(lmax=lmax, focus_dim=focus_dim, n_layers=2)
    return _launch_backward(
        grad_out.contiguous(),
        x_local.contiguous(),
        wigner_dt,
        tuple(rescale.tolist()),
        alpha.contiguous(),
        dst,
        layout,
        n_focus,
        n_head,
    )


@_flash_bwd_op.register_fake
def _(grad_out, x_local, wigner_dt, rescale, alpha, dst, lmax, n_head):
    return (
        torch.empty_like(x_local),
        torch.empty_like(wigner_dt),
        torch.empty_like(alpha),
    )


def _flash_setup(ctx, inputs, output):
    x_local, wigner_dt, rescale, alpha, order, row_ptr, dst, lmax, n_head = inputs
    ctx.save_for_backward(x_local, wigner_dt, rescale, alpha, dst)
    ctx.meta = (lmax, n_head)


def _flash_backward_rule(ctx, grad_out):
    x_local, wigner_dt, rescale, alpha, dst = ctx.saved_tensors
    lmax, n_head = ctx.meta
    grad_local, grad_wigner, grad_alpha = _flash_bwd_op(
        grad_out, x_local, wigner_dt, rescale, alpha, dst, lmax, n_head
    )
    return grad_local, grad_wigner, None, grad_alpha, None, None, None, None, None


_flash_op.register_autograd(_flash_backward_rule, setup_context=_flash_setup)


def flash_atten_aggregate(
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> Tensor:
    """Rotate back, weight by the attention softmax and reduce onto destinations.

    ``order`` and ``row_ptr`` are the destination CSR view the step builds once
    and every segment consumer shares.
    """
    return _flash_op(
        x_local, wigner_dt, rescale, alpha, order, row_ptr, dst, lmax, n_head
    )
