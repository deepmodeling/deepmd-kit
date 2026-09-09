# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""Fused cuTile SO(2) mixing stack.

The whole stack -- ``n_layers - 1`` gated layers followed by the identity final
layer -- runs inside one kernel, so neither the inter-layer activation nor the
gated-layer pre-activation reaches DRAM. The backward recovers the
pre-activation by replaying the stack from the operator's own input, which
removes the largest allocation of a SeZM inference step: the saved
pre-activation tensor is 2.11 GB per interaction block at production edge
counts.

Arithmetic
----------
Each fp32 product is evaluated as three fp16 tensor-core products with fp32
accumulation, the two-term split whose dropped cross term is ~2^-22 relative.
The head and the two tail corrections use separate accumulators, merged once per
tile; folding them into one would require scaling the head as well, which caps
the admissible activation magnitude at ``65504 / TAIL_SCALE``. Only the tails
are scaled, so the representation is valid up to the fp16 maximum.

Layout
------
``u0``   ``(F, E, ROW)``  focus-major activation produced by the rotate-and-mix
``out``  ``(E, F, ROW)``  edge-major result consumed by the aggregation
``w0``   ``(n_layers, F, M0, M0)`` ``m = 0`` block, ``(in, out)`` convention
``w1``   ``(n_layers, F, M1, M1)`` ``|m| = 1`` block
``gw``   ``(n_gated, F, Cf, lmax * Cf)`` sigmoid-gate projection

The two orientations are the layouts the neighbouring operators already use, so
neither side pays a repacking copy.

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
    split_fp16,
)
from .indexing import (
    SO2TileLayout,
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

__all__ = ["so2_mixing_stack"]

_HEADER = '''# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generated cuTile SO(2) mixing stack: lmax={lmax} Cf={cf} layers={layers} BE={be}."""

from typing import Annotated

import cuda.tile as ct

BigArray = Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64)]
BE = {be}
CF = {cf}
TAIL = {tail!r}


@ct.function
def _sigmoid(x):
    return 1.0 / (1.0 + ct.exp(-x))


@ct.function
def _contract(x, wh, wl, layer, focus, n_slab, width):
    """Return ``x @ w`` in fp32 from three fp16 tensor-core products.

    The contraction runs over ``n_slab`` real degree slabs rather than over the
    padded width: the padded weight rows are exact zeros, so skipping them is
    free. Padding survives only on the output axis, where the extent must be a
    power of two.
    """
    head = ct.zeros((BE, width), dtype=ct.float32)
    tail = ct.zeros((BE, width), dtype=ct.float32)
    for slab in range(n_slab):
        block = ct.extract(x, (0, slab), (BE, CF))
        hi = block.astype(ct.float16)
        lo = ((block - hi.astype(ct.float32)) * TAIL).astype(ct.float16)
        wh_tile = ct.reshape(
            ct.load(wh, (layer, focus, slab, 0), (1, 1, CF, width)), (CF, width)
        )
        wl_tile = ct.reshape(
            ct.load(wl, (layer, focus, slab, 0), (1, 1, CF, width)), (CF, width)
        )
        head = ct.mma(hi, wh_tile, head)
        tail = ct.mma(lo, wh_tile, tail)
        tail = ct.mma(hi, wl_tile, tail)
    return head + tail * (1.0 / TAIL)


@ct.function
def _contract_offset(x, offset, wh, wl, layer, focus, n_slab, width):
    """Return ``(x + e0 * offset) @ w``, with ``offset`` entering the first slab.

    The gate-logit gradient re-enters the backward through the scalar rows only.
    Folding it into the operand of the contraction that already reads this weight
    saves one traversal of that weight per gated layer, which is the term this
    kernel spends its time on.
    """
    head = ct.zeros((BE, width), dtype=ct.float32)
    tail = ct.zeros((BE, width), dtype=ct.float32)
    for slab in range(n_slab):
        block = ct.extract(x, (0, slab), (BE, CF))
        if slab == 0:
            block = block + offset
        hi = block.astype(ct.float16)
        lo = ((block - hi.astype(ct.float32)) * TAIL).astype(ct.float16)
        wh_tile = ct.reshape(
            ct.load(wh, (layer, focus, slab, 0), (1, 1, CF, width)), (CF, width)
        )
        wl_tile = ct.reshape(
            ct.load(wl, (layer, focus, slab, 0), (1, 1, CF, width)), (CF, width)
        )
        head = ct.mma(hi, wh_tile, head)
        tail = ct.mma(lo, wh_tile, tail)
        tail = ct.mma(hi, wl_tile, tail)
    return head + tail * (1.0 / TAIL)


@ct.function
def _gated_layer(u0, u1, w0h, w0l, w1h, w1l, g0h, g0l, g1h, g1l, layer, focus,
                 n0, n1, width0, width1):
    """Apply one gated layer to both degree groups."""
    z0 = _contract(u0, w0h, w0l, layer, focus, n0, width0)
    scalar = ct.extract(z0, (0, 0), (BE, CF))
    out0 = u0 + z0 * _sigmoid(_contract(scalar, g0h, g0l, layer, focus, 1, width0))
    z1 = _contract(u1, w1h, w1l, layer, focus, n1, width1)
    out1 = u1 + z1 * _sigmoid(_contract(scalar, g1h, g1l, layer, focus, 1, width1))
    return out0, out1
'''


def _generate(layout: SO2TileLayout, block_edges: int) -> str:
    """Return the source of the forward and backward kernels of one configuration."""
    n0, n1 = layout.n_m0, layout.n_m1
    w0, w1 = layout.width_m0, layout.width_m1
    cf, gated = layout.focus_dim, layout.n_gated
    source = [
        _HEADER.format(
            lmax=layout.lmax,
            cf=cf,
            layers=layout.n_layers,
            be=block_edges,
            tail=2048.0,
        )
    ]

    # === Forward ===
    emit = Emitter()
    emit("focus = ct.bid(1)")
    emit("edge = ct.bid(0)")
    for row in range(n0 + n1):
        emit(
            f"row{row} = ct.reshape(ct.load(uin, (focus, edge, {row}), (1, BE, CF), "
            "padding_mode=ct.PaddingMode.ZERO), (BE, CF))"
        )
    emit.concat([f"row{r}" for r in range(n0)], layout.pad_m0, "u0", "f0")
    emit.concat([f"row{n0 + r}" for r in range(n1)], layout.pad_m1, "u1", "f1")
    for layer in range(gated):
        emit(
            f"u0, u1 = _gated_layer(u0, u1, w0h, w0l, w1h, w1l, g0h, g0l, g1h, g1l,"
            f" {layer}, focus, {n0}, {n1}, {w0}, {w1})"
        )
    emit(f"u0 = u0 + _contract(u0, w0h, w0l, {gated}, focus, {n0}, {w0})")
    emit(f"u1 = u1 + _contract(u1, w1h, w1l, {gated}, focus, {n1}, {w1})")
    for row in range(n0):
        emit(
            f"ct.store(out, (edge, focus, {row}), "
            f"ct.reshape(ct.extract(u0, (0, {row}), (BE, CF)), (BE, 1, CF)))"
        )
    for row in range(n1):
        emit(
            f"ct.store(out, (edge, focus, {n0 + row}), "
            f"ct.reshape(ct.extract(u1, (0, {row}), (BE, CF)), (BE, 1, CF)))"
        )
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def stack_forward(uin: BigArray, out: BigArray,",
                "                  w0h, w0l, w1h, w1l, g0h, g0l, g1h, g1l):",
                '    """Run every layer for one edge tile of one focus stream."""',
            ],
        )
    )

    # === Backward ===
    emit = Emitter()
    emit("focus = ct.bid(1)")
    emit("edge = ct.bid(0)")
    emit("# === Replay the stack forward, keeping each layer's input ===")
    for row in range(n0 + n1):
        emit(
            f"row{row} = ct.reshape(ct.load(uin, (focus, edge, {row}), (1, BE, CF), "
            "padding_mode=ct.PaddingMode.ZERO), (BE, CF))"
        )
    emit.concat([f"row{r}" for r in range(n0)], layout.pad_m0, "a0", "b0")
    emit.concat([f"row{n0 + r}" for r in range(n1)], layout.pad_m1, "b0_", "b1")
    emit("c0 = b0_")
    for layer in range(gated):
        emit(
            f"a{layer + 1}, c{layer + 1} = _gated_layer(a{layer}, c{layer},"
            " w0h, w0l, w1h, w1l, g0h, g0l, g1h, g1l,"
            f" {layer}, focus, {n0}, {n1}, {w0}, {w1})"
        )
    emit("")
    emit("# === Gradient of the identity final layer ===")
    for row in range(n0 + n1):
        emit(
            f"grow{row} = ct.reshape(ct.load(gout, (edge, focus, {row}), (BE, 1, CF), "
            "padding_mode=ct.PaddingMode.ZERO), (BE, CF))"
        )
    emit.concat([f"grow{r}" for r in range(n0)], layout.pad_m0, "g0", "c0")
    emit.concat([f"grow{n0 + r}" for r in range(n1)], layout.pad_m1, "g1", "c1")
    emit(f"g0 = g0 + _contract(g0, t0h, t0l, {gated}, focus, {n0}, {w0})")
    emit(f"g1 = g1 + _contract(g1, t1h, t1l, {gated}, focus, {n1}, {w1})")
    emit("")
    emit("# === Gated layers in reverse over the replayed inputs ===")
    for layer in range(gated - 1, -1, -1):
        emit(f"z0 = _contract(a{layer}, w0h, w0l, {layer}, focus, {n0}, {w0})")
        emit("scalar = ct.extract(z0, (0, 0), (BE, CF))")
        emit(f"s0 = _sigmoid(_contract(scalar, g0h, g0l, {layer}, focus, 1, {w0}))")
        emit(f"z1 = _contract(c{layer}, w1h, w1l, {layer}, focus, {n1}, {w1})")
        emit(f"s1 = _sigmoid(_contract(scalar, g1h, g1l, {layer}, focus, 1, {w1}))")
        emit("dlogit0 = (g0 * z0) * s0 * (1.0 - s0)")
        emit("dlogit1 = (g1 * z1) * s1 * (1.0 - s1)")
        emit(
            f"dscalar = _contract(dlogit0, q0h, q0l, {layer}, focus, {n0}, CF)"
            f" + _contract(dlogit1, q1h, q1l, {layer}, focus, {n1}, CF)"
        )
        emit(
            f"gnext = g0 + _contract_offset(g0 * s0, dscalar, t0h, t0l,"
            f" {layer}, focus, {n0}, {w0})"
        )
        emit(f"g1 = g1 + _contract(g1 * s1, t1h, t1l, {layer}, focus, {n1}, {w1})")
        emit("g0 = gnext")
    for row in range(n0):
        emit(
            f"ct.store(gin, (focus, edge, {row}), "
            f"ct.reshape(ct.extract(g0, (0, {row}), (BE, CF)), (1, BE, CF)))"
        )
    for row in range(n1):
        emit(
            f"ct.store(gin, (focus, edge, {n0 + row}), "
            f"ct.reshape(ct.extract(g1, (0, {row}), (BE, CF)), (1, BE, CF)))"
        )
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def stack_backward(uin: BigArray, gout: BigArray, gin: BigArray,",
                "                   w0h, w0l, w1h, w1l, g0h, g0l, g1h, g1l,",
                "                   t0h, t0l, t1h, t1l, q0h, q0l, q1h, q1l):",
                '    """Gradient of the stack with respect to its input activation."""',
            ],
        )
    )
    return "".join(source)


def _module(layout: SO2TileLayout, block_edges: int) -> ModuleType:
    stem = f"sezm_stack_l{layout.lmax}_c{layout.focus_dim}_n{layout.n_layers}_b{block_edges}"
    return generated_module(stem, _generate(layout, block_edges))


def pack_weights(
    w0: Tensor, w1: Tensor, gw: Tensor, layout: SO2TileLayout
) -> dict[str, Tensor]:
    """Pad the stack weights to power-of-two degree groups and split them to fp16.

    The gate projection is expanded into one matrix per degree group whose column
    blocks already carry the degree-to-gate mapping: block 0 of the ``m = 0``
    projection is the identity, so its sigmoid is the SiLU gate of the scalar
    rows, block ``r`` is gate ``r - 1``, and the ``|m| = 1`` projection replicates
    gate ``o mod lmax``. One contraction per group then produces a full-width gate
    tile that multiplies the pre-activation elementwise, which avoids a scatter
    into a column block -- an operation the tile model does not provide.

    Parameters
    ----------
    w0, w1 : Tensor
        Per-layer ``m = 0`` and ``|m| = 1`` blocks, ``(n_layers, F, M, M)``.
    gw : Tensor
        Gate projection, ``(n_gated, F, Cf, lmax * Cf)``.
    layout : SO2TileLayout
        Configuration geometry.

    Returns
    -------
    dict[str, Tensor]
        fp16 head and tail of the forward weights (``w0``, ``w1``), the gate
        projections (``g0``, ``g1``), and the transposes the backward needs
        (``t0``, ``t1``, ``q0``, ``q1``).
    """
    n_focus = w0.shape[1]
    cf = layout.focus_dim
    device = w0.device

    padded0 = torch.nn.functional.pad(w0, (0, layout.width_m0 - layout.n_m0 * cf) * 2)
    padded1 = torch.nn.functional.pad(w1, (0, layout.width_m1 - layout.n_m1 * cf) * 2)

    gate0 = gw.new_zeros(layout.n_gated, n_focus, cf, layout.width_m0)
    gate0[:, :, :, :cf] = torch.eye(cf, device=device, dtype=gw.dtype)
    gate0[:, :, :, cf : layout.n_m0 * cf] = gw
    gate1 = gw.new_zeros(layout.n_gated, n_focus, cf, layout.width_m1)
    # Degree ``o`` of the ``|m| = 1`` group takes gate ``o mod lmax``, and that
    # group holds exactly two degrees per order, so the mapping is the gate
    # projection repeated once.
    gate1[:, :, :, : layout.n_m1 * cf] = gw.repeat(1, 1, 1, 2)

    packed: dict[str, Tensor] = {}
    for name, tensor in (
        ("w0", padded0),
        ("w1", padded1),
        ("g0", gate0),
        ("g1", gate1),
    ):
        packed[name + "h"], packed[name + "l"] = split_fp16(tensor)
    # The backward reads the same four matrices transposed. Narrowing commutes
    # with transposition elementwise, so the transposed halves are the transposes
    # of the halves -- bit-identical, at half the number of splits.
    for source, target in (("w0", "t0"), ("w1", "t1"), ("g0", "q0"), ("g1", "q1")):
        for half in ("h", "l"):
            packed[target + half] = packed[source + half].transpose(-1, -2).contiguous()
    return packed


def _launch_forward(
    u0: Tensor, packed: dict[str, Tensor], layout: SO2TileLayout
) -> Tensor:
    """Run the fused stack and return the edge-major ``(E, F, ROW)`` activation."""
    n_focus, n_edge, row = u0.shape
    out = u0.new_empty((n_edge, n_focus, row))
    config = tile_config("mixing_stack_fwd", layout.key)
    module = _module(layout, config.tile)
    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(n_edge / config.tile), n_focus),
        kernel_variant(module.stack_forward, **config.hints),
        (
            u0,
            out,
            packed["w0h"],
            packed["w0l"],
            packed["w1h"],
            packed["w1l"],
            packed["g0h"],
            packed["g0l"],
            packed["g1h"],
            packed["g1l"],
        ),
    )
    return out


def _launch_backward(
    u0: Tensor, grad_out: Tensor, packed: dict[str, Tensor], layout: SO2TileLayout
) -> Tensor:
    """Return the gradient of the fused stack with respect to its input."""
    n_focus, n_edge, _ = u0.shape
    grad_in = torch.empty_like(u0)
    config = tile_config("mixing_stack_bwd", layout.key)
    module = _module(layout, config.tile)
    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(n_edge / config.tile), n_focus),
        kernel_variant(module.stack_backward, **config.hints),
        (
            u0,
            grad_out,
            grad_in,
            packed["w0h"],
            packed["w0l"],
            packed["w1h"],
            packed["w1l"],
            packed["g0h"],
            packed["g0l"],
            packed["g1h"],
            packed["g1l"],
            packed["t0h"],
            packed["t0l"],
            packed["t1h"],
            packed["t1l"],
            packed["q0h"],
            packed["q0l"],
            packed["q1h"],
            packed["q1l"],
        ),
    )
    return grad_in


@torch.library.custom_op("sezm_cutile::mixing_stack", mutates_args=())
def _stack_op(
    u0: Tensor,
    w0: Tensor,
    w1: Tensor,
    gw: Tensor,
    lmax: int,
    focus_dim: int,
) -> Tensor:
    layout = SO2TileLayout(lmax=lmax, focus_dim=focus_dim, n_layers=w0.shape[0])
    return _launch_forward(u0.contiguous(), pack_weights(w0, w1, gw, layout), layout)


@_stack_op.register_fake
def _(u0, w0, w1, gw, lmax, focus_dim):
    n_focus, n_edge, row = u0.shape
    return u0.new_empty((n_edge, n_focus, row))


@torch.library.custom_op("sezm_cutile::mixing_stack_bwd", mutates_args=())
def _stack_bwd_op(
    u0: Tensor,
    grad_out: Tensor,
    w0: Tensor,
    w1: Tensor,
    gw: Tensor,
    lmax: int,
    focus_dim: int,
) -> Tensor:
    layout = SO2TileLayout(lmax=lmax, focus_dim=focus_dim, n_layers=w0.shape[0])
    return _launch_backward(
        u0.contiguous(),
        grad_out.contiguous(),
        pack_weights(w0, w1, gw, layout),
        layout,
    )


@_stack_bwd_op.register_fake
def _(u0, grad_out, w0, w1, gw, lmax, focus_dim):
    return torch.empty_like(u0)


def _stack_setup(ctx, inputs, output):
    u0, w0, w1, gw, lmax, focus_dim = inputs
    ctx.save_for_backward(u0, w0, w1, gw)
    ctx.meta = (lmax, focus_dim)


def _stack_backward_rule(ctx, grad_out):
    u0, w0, w1, gw = ctx.saved_tensors
    lmax, focus_dim = ctx.meta
    grad_u0 = _stack_bwd_op(u0, grad_out, w0, w1, gw, lmax, focus_dim)
    return grad_u0, None, None, None, None, None


_stack_op.register_autograd(_stack_backward_rule, setup_context=_stack_setup)


def so2_mixing_stack(
    u0: Tensor, w0: Tensor, w1: Tensor, gw: Tensor, lmax: int, focus_dim: int
) -> Tensor:
    """Run the gated SO(2) mixing stack and return the edge-major activation."""
    return _stack_op(u0, w0, w1, gw, lmax, focus_dim)
