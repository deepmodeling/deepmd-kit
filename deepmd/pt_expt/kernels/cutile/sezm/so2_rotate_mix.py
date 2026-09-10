# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""Fused cuTile rotate-to-local and radial degree mixing.

One kernel per edge tile gathers the source node features, applies the
block-diagonal Wigner rotation into the edge-aligned frame over the structural
non-zeros only, applies the edge-conditioned radial degree mixing, and writes the
focus-major activation the mixing stack consumes. The rotated pre-mix
intermediate is never materialized.

The backward recomputes the rotation from the operator inputs -- the forward
saves nothing -- and reduces onto source nodes inside the same kernel.

Arithmetic and parallel layout
------------------------------
Neither the rotation nor the mixing is a matrix product with a shared operand:
the rotation coefficient and the mixing kernel are per-edge scalars broadcast
over channels, so both are elementwise tile arithmetic and neither uses a tensor
core. Together they are about 1.5 % of the mixing stack's multiply-adds.

The forward grid is one dimensional over edges. The backward grid is one block
per *source* node walking that node's CSR segment, which is what makes the
node-level reduction fit in the same kernel: every edge of a segment shares one
source, so its features are read once rather than once per edge, and the node
gradient accumulates in registers. The alternative -- an edge-major backward
followed by a separate segmented reduction -- writes and reads back a per-edge
intermediate the size of the node table times the mean degree, and measured 2.6
times slower end to end for this stage.

In both directions the focus streams are a compile-time loop inside the kernel
rather than a grid axis. As a grid axis they would race in the backward: the
rotation and degree-mixing gradients are shared across focus streams, so every
stream would write the same output element. Carrying them in registers across an
unrolled loop keeps the reduction exact and local.

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
from .flash_atten import (
    build_row_ptr,
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

__all__ = ["so2_rotate_mix"]

_HEADER = '''# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generated cuTile rotate-and-mix: lmax={lmax} Cf={cf} F={focus} BE={be}."""

from typing import Annotated

import cuda.tile as ct

BigArray = Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64)]
BE = {be}
BS = {bs}
CF = {cf}
CW = {cw}
NODE_STRIDE = {node_stride}
'''


def _generate(
    layout: SO2TileLayout, n_focus: int, block_edges: int, segment_edges: int
) -> str:
    """Return the source of the rotate-and-mix forward and backward kernels."""
    cf = layout.focus_dim
    c_wide = n_focus * cf
    n0, lmax = layout.n_m0, layout.lmax
    n_row = n0 + layout.n_m1
    pairs = rotation_pairs(lmax)
    coeff = m_major_index(lmax)
    by_row: dict[int, list[tuple[int, int]]] = {}
    by_full: dict[int, list[tuple[int, int]]] = {}
    for slot, (reduced, full) in enumerate(pairs):
        by_row.setdefault(reduced, []).append((slot, full))
        by_full.setdefault(full, []).append((slot, reduced))
    full_rows = sorted(by_full)

    def emit_coefficients(emit: Emitter) -> None:
        """Load the structural non-zeros of the reduced rotation.

        One scalar load per coefficient outperforms a single coalesced load of
        the whole padded row followed by extraction, by about 25 % on both
        directions: the rotation blocks are small enough to stay in cache, so the
        wider tile buys nothing and costs registers.
        """
        for slot, (reduced, full) in enumerate(pairs):
            emit(
                f"d{slot} = ct.reshape(ct.load(wigner, (edge, {coeff[reduced]},"
                f" {full}), (BE, 1, 1), padding_mode=ct.PaddingMode.ZERO), (BE, 1))"
            )

    def emit_mixer(emit: Emitter) -> None:
        """Load the compact per-edge degree-mixing kernel."""
        for slot in range(layout.kernel_size):
            emit(
                f"k{slot} = ct.reshape(ct.load(mixer, (edge, {slot}), (BE, 1),"
                " padding_mode=ct.PaddingMode.ZERO), (BE, 1))"
            )

    def emit_rotation(emit: Emitter, focus: int) -> None:
        """Gather one focus stream's source rows and project them onto the reduced rows."""
        emit(f"lane = {focus * cf} + ct.arange(CF, dtype=ct.int32).reshape((1, CF))")
        emit(f"basis = ct.reshape(ct.load(channel, ({focus},), (CF,)), (1, CF))")
        for full in full_rows:
            emit(
                f"x{full} = ct.gather(xnode, base + {full * c_wide} + lane,"
                " check_bounds=False)"
            )
        for reduced in range(n_row):
            emit(
                f"r{reduced} = "
                + " + ".join(f"d{slot} * x{full}" for slot, full in by_row[reduced])
            )

    def mix_group(
        prefix: str, source: str, offset: int, count: int, base: int
    ) -> list[str]:
        """Return the degree-mixing statements of one ``|m|`` group.

        The compact kernel is indexed ``[input_degree, output_degree]``, so the
        forward contracts over the input degree.
        """
        return [
            f"{prefix}{offset + out} = basis * ("
            + " + ".join(
                f"k{base + inp * count + out} * {source}{offset + inp}"
                for inp in range(count)
            )
            + ")"
            for out in range(count)
        ]

    def mix_group_transposed(
        prefix: str, source: str, offset: int, count: int, base: int
    ) -> list[str]:
        """Return the adjoint of :func:`mix_group`, contracting the output degree."""
        return [
            f"{prefix}{offset + inp} = basis * ("
            + " + ".join(
                f"k{base + inp * count + out} * {source}{offset + out}"
                for out in range(count)
            )
            + ")"
            for inp in range(count)
        ]

    source = [
        _HEADER.format(
            lmax=lmax,
            cf=cf,
            focus=n_focus,
            be=block_edges,
            bs=segment_edges,
            cw=c_wide,
            node_stride=layout.dim * c_wide,
        )
    ]

    # === Forward ===
    emit = Emitter()
    emit("edge = ct.bid(0)")
    emit("srcn = ct.load(srcs, (edge,), (BE,), padding_mode=ct.PaddingMode.ZERO)")
    emit("base = srcn.reshape((BE, 1)) * NODE_STRIDE")
    emit_coefficients(emit)
    emit_mixer(emit)
    for focus in range(n_focus):
        emit("")
        emit(f"# === Focus stream {focus} ===")
        emit_rotation(emit, focus)
        emit.extend(mix_group("y", "r", 0, n0, 0))
        emit.extend(mix_group("y", "r", n0, lmax, n0 * n0))
        emit.extend(mix_group("y", "r", n0 + lmax, lmax, n0 * n0))
        for reduced in range(n_row):
            emit(
                f"ct.store(out, ({focus}, edge, {reduced}),"
                f" ct.reshape(y{reduced}, (1, BE, CF)))"
            )
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def rotate_mix_forward(xnode: BigArray, srcs, wigner: BigArray,",
                "                       mixer: BigArray, channel, out: BigArray):",
                '    """Rotate, mix and store one edge tile."""',
            ],
        )
    )

    # === Backward ===
    emit = Emitter()
    emit("node = ct.bid(0)")
    emit("start = ct.load(row_ptr, (node,), (1,)).item()")
    emit("stop = ct.load(row_ptr, (node + 1,), (1,)).item()")
    for focus in range(n_focus):
        emit(f"basis{focus} = ct.reshape(ct.load(channel, ({focus},), (CF,)), (1, CF))")
        for full in full_rows:
            emit(
                f"xn{focus}_{full} = ct.reshape(ct.load(xnode, (node, {full},"
                f" {focus}, 0), (1, 1, 1, CF)), (1, CF))"
            )
            emit(f"acc{focus}_{full} = ct.zeros((CF,), dtype=ct.float32)")
    emit("for position in range(start, stop, BS):")
    walk = Emitter(indent="        ")
    walk("slot = position + ct.arange(BS, dtype=ct.int32)")
    walk("live = slot < stop")
    walk("entry = ct.gather(order, ct.where(live, slot, stop - 1), check_bounds=False)")
    walk("keep = ct.where(live.reshape((BS, 1)), 1.0, 0.0)")
    # Lanes past the end of a segment are redirected to a scratch row appended to
    # every per-edge output, so a masked store needs no predication support. The
    # scratch index is broadcast into a tile by arithmetic and then narrowed
    # explicitly: selecting directly between operands of different integer widths
    # emits a scalar narrowing that fails tile-compiler verification.
    walk("sink = ct.where(live, entry, (entry * 0 + n_edge).astype(ct.int32))")
    for slot, (reduced, full) in enumerate(pairs):
        walk(
            f"d{slot} = ct.load_advanced_indexing(wigner, (entry,"
            f" ct.Slice({coeff[reduced] * layout.dim + full}, 1)),"
            " padding_mode=ct.PaddingMode.ZERO)"
        )
    for slot in range(layout.kernel_size):
        walk(
            f"k{slot} = ct.load_advanced_indexing(mixer, (entry,"
            f" ct.Slice({slot}, 1)), padding_mode=ct.PaddingMode.ZERO)"
        )
    for slot in range(len(pairs)):
        walk(f"gd{slot} = ct.zeros((BS,), dtype=ct.float32)")
    for slot in range(layout.kernel_size):
        walk(f"gk{slot} = ct.zeros((BS,), dtype=ct.float32)")
    for focus in range(n_focus):
        walk("")
        walk(f"# === Focus stream {focus}: replay, then differentiate ===")
        for reduced in range(n_row):
            walk(
                f"r{reduced} = "
                + " + ".join(
                    f"d{slot} * xn{focus}_{full}" for slot, full in by_row[reduced]
                )
            )
            walk(
                f"g{reduced} = ct.load_advanced_indexing(gout, (entry,"
                f" ct.Slice({(focus * n_row + reduced) * cf}, CF)),"
                " padding_mode=ct.PaddingMode.ZERO) * keep"
            )
        # The mixing is linear in both operands: the kernel gradient is the
        # channel-summed product of the rotated row with the output gradient,
        # and the rotated-row gradient is the kernel-weighted output gradient.
        # Both accumulate across focus streams because both operands are shared.
        for out in range(n0):
            for inp in range(n0):
                walk(
                    f"gk{inp * n0 + out} = gk{inp * n0 + out}"
                    f" + ct.sum(r{inp} * basis{focus} * g{out}, axis=1)"
                )
        for out in range(lmax):
            for inp in range(lmax):
                slot = n0 * n0 + inp * lmax + out
                neg, pos = n0, n0 + lmax
                walk(
                    f"gk{slot} = gk{slot}"
                    f" + ct.sum(r{neg + inp} * basis{focus} * g{neg + out}, axis=1)"
                    f" + ct.sum(r{pos + inp} * basis{focus} * g{pos + out}, axis=1)"
                )
        walk.extend(
            statement.replace("basis", f"basis{focus}")
            for statement in mix_group_transposed("h", "g", 0, n0, 0)
            + mix_group_transposed("h", "g", n0, lmax, n0 * n0)
            + mix_group_transposed("h", "g", n0 + lmax, lmax, n0 * n0)
        )
        for slot, (reduced, full) in enumerate(pairs):
            walk(f"gd{slot} = gd{slot} + ct.sum(h{reduced} * xn{focus}_{full}, axis=1)")
        for full in full_rows:
            terms = " + ".join(
                f"d{slot} * h{reduced}" for slot, reduced in by_full[full]
            )
            walk(f"acc{focus}_{full} = acc{focus}_{full} + ct.sum({terms}, axis=0)")
    walk("")
    walk("# === Per-edge gradients, complete once every focus stream is folded in ===")
    for slot, (reduced, full) in enumerate(pairs):
        walk(
            f"ct.store_advanced_indexing(gwigner, (sink,"
            f" ct.Slice({coeff[reduced] * layout.dim + full}, 1)),"
            f" ct.reshape(gd{slot}, (BS, 1)))"
        )
    for slot in range(layout.kernel_size):
        walk(
            f"ct.store_advanced_indexing(gmixer, (sink, ct.Slice({slot}, 1)),"
            f" ct.reshape(gk{slot}, (BS, 1)))"
        )
    emit.extend([line[4:] for line in walk.lines])
    emit("")
    for focus in range(n_focus):
        for full in range(layout.dim):
            value = (
                f"acc{focus}_{full}"
                if full in by_full
                else "ct.zeros((CF,), dtype=ct.float32)"
            )
            emit(
                f"ct.store(gx, (node, {full}, {focus}, 0),"
                f" ct.reshape({value}, (1, 1, 1, CF)))"
            )
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def rotate_mix_backward(xnode: BigArray, order, row_ptr,",
                "                        wigner: BigArray, mixer: BigArray, channel,",
                "                        gout: BigArray, gx: BigArray,",
                "                        gwigner: BigArray, gmixer: BigArray,",
                "                        n_edge: ct.ScalarInt64):",
                '    """Differentiate one source node\'s edges and reduce onto it.',
                "",
                "        The walk is source major so every edge of a segment shares one",
                "        source node: its features are read once instead of once per",
                "        edge, and the node gradient is reduced in registers, which",
                "        removes the per-edge intermediate the separate segmented",
                "        reduction would otherwise have to write and read back.",
                '        """',
            ],
        )
    )
    return "".join(source)


def _edge_tile(layout: SO2TileLayout) -> int:
    """Return the forward edge tile, which also sizes the generated module."""
    return tile_config("rotate_mix_fwd", layout.key).tile


def _segment_tile(layout: SO2TileLayout) -> int:
    """Return the backward segment tile, which also sizes the generated module."""
    return tile_config("rotate_mix_bwd", layout.key).tile


def _module(
    layout: SO2TileLayout, n_focus: int, block_edges: int, segment_edges: int
) -> ModuleType:
    stem = (
        f"sezm_rotmix_l{layout.lmax}_c{layout.focus_dim}_f{n_focus}"
        f"_b{block_edges}_s{segment_edges}"
    )
    return generated_module(
        stem, _generate(layout, n_focus, block_edges, segment_edges)
    )


def _launch_forward(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    mixer: Tensor,
    channel: Tensor,
    layout: SO2TileLayout,
    n_focus: int,
) -> Tensor:
    """Rotate the source features into the edge frame and apply the degree mixing.

    Parameters
    ----------
    x : Tensor
        Node features, ``(N, D, C_wide)``.
    src : Tensor
        Source node of each edge, ``(E,)``.
    wigner : Tensor
        Block-diagonal Wigner-D per edge, ``(E, D, D)``.
    mixer : Tensor
        Compact per-edge degree-mixing kernel, ``(E, kernel_size)``.
    channel : Tensor
        Channel basis of the mixer, ``(C_wide,)``.
    layout : SO2TileLayout
        Configuration geometry.
    n_focus : int
        Number of focus streams.

    Returns
    -------
    Tensor
        Focus-major activation ``(F, E, ROW)``.
    """
    n_edge = src.shape[0]
    out = x.new_empty((n_focus, n_edge, layout.row))
    config = tile_config("rotate_mix_fwd", layout.key)
    module = _module(layout, n_focus, config.tile, _segment_tile(layout))
    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(n_edge / config.tile),),
        kernel_variant(module.rotate_mix_forward, **config.hints),
        (x.reshape(-1), src.to(torch.int32), wigner, mixer, channel, out),
    )
    return out


def _launch_backward(
    grad_out: Tensor,
    x: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    wigner: Tensor,
    mixer: Tensor,
    channel: Tensor,
    layout: SO2TileLayout,
    n_focus: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the node feature, rotation and degree-mixing gradients.

    Parameters
    ----------
    grad_out : Tensor
        Gradient of the focus-major activation, ``(F, E, ROW)``.
    x : Tensor
        Node features, ``(N, D, C_wide)``.
    order : Tensor
        Edge indices sorted by source node, ``(E,)``.
    row_ptr : Tensor
        Source CSR offsets, ``(N + 1,)``.
    wigner, mixer, channel : Tensor
        The forward operands.
    layout : SO2TileLayout
        Configuration geometry.
    n_focus : int
        Number of focus streams.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Node gradient ``(N, D, C_wide)``, rotation gradient ``(E, D, D)`` on its
        structural support, and degree-mixing gradient ``(E, kernel_size)``.
    """
    n_edge = mixer.shape[0]
    n_node = row_ptr.shape[0] - 1
    grad_x = x.new_empty((n_node, layout.dim, n_focus, layout.focus_dim))
    # One scratch row absorbs the writes of the lanes that overrun a segment.
    grad_wigner = wigner.new_zeros((n_edge + 1, layout.dim * layout.dim))
    grad_mixer = mixer.new_empty((n_edge + 1, layout.kernel_size))
    config = tile_config("rotate_mix_bwd", layout.key)
    module = _module(layout, n_focus, _edge_tile(layout), config.tile)
    ct.launch(
        torch.cuda.current_stream(),
        (n_node,),
        kernel_variant(module.rotate_mix_backward, **config.hints),
        (
            x.view(x.shape[0], layout.dim, n_focus, layout.focus_dim),
            order.to(torch.int32),
            row_ptr.to(torch.int32),
            wigner.reshape(n_edge, -1),
            mixer,
            channel,
            grad_out.permute(1, 0, 2).reshape(n_edge, -1),
            grad_x,
            grad_wigner,
            grad_mixer,
            n_edge,
        ),
    )
    return (
        grad_x.reshape(n_node, layout.dim, n_focus * layout.focus_dim),
        grad_wigner[:n_edge].view_as(wigner),
        grad_mixer[:n_edge],
    )


@torch.library.custom_op("sezm_cutile::rotate_mix", mutates_args=())
def _rotate_mix_op(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    mixer: Tensor,
    channel: Tensor,
    lmax: int,
    focus_dim: int,
    n_focus: int,
) -> Tensor:
    layout = SO2TileLayout(lmax=lmax, focus_dim=focus_dim, n_layers=2)
    return _launch_forward(
        x.contiguous(),
        src,
        wigner,
        mixer.contiguous(),
        channel.contiguous(),
        layout,
        n_focus,
    )


@_rotate_mix_op.register_fake
def _(x, src, wigner, mixer, channel, lmax, focus_dim, n_focus):
    return x.new_empty((n_focus, src.shape[0], (3 * lmax + 1) * focus_dim))


@torch.library.custom_op("sezm_cutile::rotate_mix_bwd", mutates_args=())
def _rotate_mix_bwd_op(
    grad_out: Tensor,
    x: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    wigner: Tensor,
    mixer: Tensor,
    channel: Tensor,
    lmax: int,
    focus_dim: int,
    n_focus: int,
) -> tuple[Tensor, Tensor, Tensor]:
    layout = SO2TileLayout(lmax=lmax, focus_dim=focus_dim, n_layers=2)
    return _launch_backward(
        grad_out,
        x.contiguous(),
        order,
        row_ptr,
        wigner,
        mixer.contiguous(),
        channel.contiguous(),
        layout,
        n_focus,
    )


@_rotate_mix_bwd_op.register_fake
def _(grad_out, x, order, row_ptr, wigner, mixer, channel, lmax, focus_dim, n_focus):
    return (
        torch.empty_like(x),
        torch.empty_like(wigner),
        torch.empty_like(mixer),
    )


def _rotate_mix_setup(ctx, inputs, output):
    x, src, wigner, mixer, channel, lmax, focus_dim, n_focus = inputs
    ctx.save_for_backward(x, src, wigner, mixer, channel)
    ctx.meta = (lmax, focus_dim, n_focus)


def _rotate_mix_backward_rule(ctx, grad_out):
    x, src, wigner, mixer, channel = ctx.saved_tensors
    lmax, focus_dim, n_focus = ctx.meta
    # The backward walks source segments, so it needs the topology the forward
    # does not: sorting the edges by source costs far less than the per-edge
    # intermediate a scatter-based reduction would materialize.
    order = torch.argsort(src)
    row_ptr = build_row_ptr(src.index_select(0, order), x.shape[0])
    grad_x, grad_wigner, grad_mixer = _rotate_mix_bwd_op(
        grad_out, x, order, row_ptr, wigner, mixer, channel, lmax, focus_dim, n_focus
    )
    return grad_x, None, grad_wigner, grad_mixer, None, None, None, None


_rotate_mix_op.register_autograd(
    _rotate_mix_backward_rule, setup_context=_rotate_mix_setup
)


def so2_rotate_mix(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    mixer: Tensor,
    channel: Tensor,
    lmax: int,
    focus_dim: int,
    n_focus: int,
) -> Tensor:
    """Rotate the source features into the edge frame and mix the degrees."""
    return _rotate_mix_op(x, src, wigner, mixer, channel, lmax, focus_dim, n_focus)
