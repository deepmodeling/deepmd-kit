# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused DPA1 graph-lower descriptor operator.

The CUDA operator ``deepmd::dpa1_graph_descriptor`` (see
``source/op/pt/dpa1_graph_descriptor.cu``) evaluates the whole DPA1
(``se_atten``) attention-free descriptor body from the flat edge stream --
environment matrix, three-layer embedding MLP, the strip type-pair gate when
applicable, moment reduction and the ``G^T G`` contraction -- in two mega
kernels (forward and backward). This module supplies everything the pt_expt
graph lower needs beyond the raw CUDA implementation:

* **Fake (meta) implementations** so ``make_fx`` / ``torch.export`` can infer
  output shapes and bake the operator into the graph-form ``.pt2`` without
  running CUDA.
* **A registered backward** routing to
  ``deepmd::dpa1_graph_descriptor_backward``. The graph lower assembles force
  and virial analytically from ``grad(E, edge_vec)``, so the descriptor must
  expose an ``edge_vec`` gradient; weights and integer topology are constants
  of the inference graph and receive ``None``.
* **CPU reference implementations** registered on the same ops. The pt_expt
  freeze pipeline traces and sample-evaluates the model on CPU; these keep the
  ops functional there (trace-time only -- deployment always dispatches the
  CUDA kernels).

Tebd input modes
----------------
Concat folds the type embedding into the layer-1 input; the operator receives
an empty ``gate_table``. Strip runs the embedding on the radial channel alone
and multiplies the output by the type-pair gate
``1 + embeddings_strip(tebd_pair) (* sw when smooth_type_embedding)``; the
wrapper precomputes the gate table (``(T or T^2, ng)``) from the strip
network, so the operator never evaluates it per edge.

Usage and pitfalls
------------------
* The forward returns ``(grrg, rot_mat)`` plus five auxiliary tensors saved
  for the backward (moment, CSR edge order, folded layer-1 pair table, and
  the transposed ``pre2`` / ``g`` activation spills). The auxiliaries never
  receive gradients; ``set_materialize_grads(False)`` skips their zero fill.
* The activation spills cost ``(2 * neuron[1] + neuron[2]) * 4`` bytes per
  edge (768 B/edge for the (32, 64, 128) net). This is far below the autograd
  tape of the reference path, but at tens of millions of edges it is
  gigabytes; the buffers live only between forward and backward.
* Weight tensors must be fp32 and contiguous; ``edge_vec`` may be fp32 or
  fp64 (the backward returns the gradient in the leaf dtype and the internal
  compute is fp32 either way). The caller-facing gate lives in
  :meth:`DescrptDPA1._fused_eligible`.
* Constants captured by the op call (weights, statistics slices, the strip
  gate table) are prepared per call rather than cached on the module: values
  produced while a fake / proxy tensor mode is active must never leak into
  eager state.
"""

from typing import (
    Any,
)

import torch

from deepmd.kernels.triton.dpa1.activation import (
    ACT_CODES,
)

__all__ = [
    "dpa1_graph_descriptor",
    "ensure_registered",
    "op_available",
]

_TILE_EDGES = 128  # matches kTileEdges in dpa1_graph_descriptor.cu


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa1_graph_descriptor`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa1_graph_descriptor", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


# ======================================================================
# Fake (meta) implementations
# ======================================================================
def _forward_fake(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    davg: torch.Tensor,
    dstd: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    idt1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    idt2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    idt3: torch.Tensor,
    gate_table: torch.Tensor,
    act: int,
    type_one_side: int,
    concat_tebd: int,
    write_rotation: int,
    smooth: int,
    axis: int,
    resnet2: int,
    resnet3: int,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
) -> tuple[torch.Tensor, ...]:
    n_edge = edge_vec.shape[0]
    n_node = atype.shape[0]
    n2 = w2.shape[1]
    ng = w3.shape[1]
    ntypes, tebd_dim = type_embedding.shape
    out_dim = ng * axis + (tebd_dim if concat_tebd else 0)
    # Strip mode carries the type pairs in the gate table; the layer-1 pair
    # table degenerates to its single bias row.
    strip = gate_table.shape[0] > 0
    n_pairs = 1 if strip else (ntypes if type_one_side else ntypes * ntypes)
    e_pad = (n_edge + _TILE_EDGES - 1) // _TILE_EDGES * _TILE_EDGES
    dev = edge_vec.device
    return (
        torch.empty(n_node, out_dim, dtype=torch.float32, device=dev),
        torch.empty(
            n_node if write_rotation else 0,
            ng,
            3,
            dtype=torch.float32,
            device=dev,
        ),
        torch.empty(n_node, 4, ng, dtype=torch.float32, device=dev),
        torch.empty(n_edge, dtype=torch.int32, device=dev),
        torch.empty(n_pairs, w1.shape[1], dtype=torch.float32, device=dev),
        torch.empty(n2, e_pad, dtype=torch.float32, device=dev),
        torch.empty(ng, e_pad, dtype=torch.float32, device=dev),
    )


def _backward_fake(
    d_grrg: torch.Tensor,
    d_rot_mat: torch.Tensor | None,
    gr: torch.Tensor,
    edge_order: torch.Tensor,
    pair_table: torch.Tensor,
    pre2_saved: torch.Tensor,
    g_saved: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    return torch.empty_like(edge_vec)


# ======================================================================
# Autograd bridge
# ======================================================================
def _setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    (
        edge_vec,
        edge_index,
        edge_mask,
        atype,
        type_embedding,
        davg,
        dstd,
        w1,
        b1,
        idt1,
        w2,
        b2,
        idt2,
        w3,
        b3,
        idt3,
        gate_table,
        act,
        type_one_side,
        concat_tebd,
        _write_rotation,
        smooth,
        axis,
        resnet2,
        resnet3,
        rcut,
        rcut_smth,
        protection,
        nnei,
    ) = inputs
    # gr, edge_order, pair_table, pre2_saved, g_saved; the type embedding is
    # not saved -- the backward re-reads layer 1 through the folded pair table
    # and the strip gate through the gate table.
    aux = output[2:]
    ctx.save_for_backward(
        *aux,
        edge_vec,
        edge_index,
        edge_mask,
        atype,
        davg,
        dstd,
        w1,
        b1,
        idt1,
        w2,
        b2,
        idt2,
        w3,
        b3,
        idt3,
        gate_table,
    )
    ctx.n_aux = len(aux)
    ctx.scalars = (
        act,
        type_one_side,
        smooth,
        axis,
        resnet2,
        resnet3,
        rcut,
        rcut_smth,
        protection,
        nnei,
    )
    # The auxiliary outputs never receive real gradients; skipping their zero
    # materialization saves several full-size fills per step.
    ctx.set_materialize_grads(False)


def _backward(
    ctx: Any,
    d_grrg: torch.Tensor,
    d_rot_mat: torch.Tensor | None,
    *d_aux: Any,
) -> tuple:
    saved = ctx.saved_tensors
    aux = saved[: ctx.n_aux]
    (
        edge_vec,
        edge_index,
        edge_mask,
        atype,
        davg,
        dstd,
        w1,
        b1,
        idt1,
        w2,
        b2,
        idt2,
        w3,
        b3,
        idt3,
        gate_table,
    ) = saved[ctx.n_aux :]
    (
        act,
        type_one_side,
        smooth,
        axis,
        resnet2,
        resnet3,
        rcut,
        rcut_smth,
        protection,
        nnei,
    ) = ctx.scalars
    # rot_mat is unused by the energy fitting; autograd then hands None, which
    # the op schema accepts (Tensor? d_rot_mat).
    d_edge_vec = torch.ops.deepmd.dpa1_graph_descriptor_backward(
        d_grrg,
        d_rot_mat,
        *aux,
        edge_vec,
        edge_index,
        edge_mask,
        atype,
        davg,
        dstd,
        w1,
        b1,
        idt1,
        w2,
        b2,
        idt2,
        w3,
        b3,
        idt3,
        gate_table,
        act,
        type_one_side,
        smooth,
        axis,
        resnet2,
        resnet3,
        rcut,
        rcut_smth,
        protection,
        nnei,
    )
    return (d_edge_vec,) + (None,) * 28


# ======================================================================
# CPU reference implementations
# ======================================================================
def _act(z: torch.Tensor, act: int) -> torch.Tensor:
    return torch.tanh(z) if act == 0 else torch.nn.functional.silu(z)


def _cpu_pair_table(
    type_embedding: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    type_one_side: int,
    strip: bool,
) -> torch.Tensor:
    """Fold the type-embedding rows of W1 (and b1) into the per-pair table.

    Strip mode has no type term in layer 1: the table is the single bias row.
    """
    n1 = w1.shape[1]
    bias = b1 if b1.numel() else torch.zeros(n1, dtype=w1.dtype, device=w1.device)
    if strip:
        return bias.reshape(1, n1).contiguous()
    ntypes, tebd_dim = type_embedding.shape
    nei = type_embedding @ w1[1 : 1 + tebd_dim]  # (ntypes, n1)
    if type_one_side:
        return (nei + bias).contiguous()
    center = type_embedding @ w1[1 + tebd_dim : 1 + 2 * tebd_dim]
    return (
        (center[:, None, :] + nei[None, :, :] + bias)
        .reshape(ntypes * ntypes, -1)
        .contiguous()
    )


def _cpu_layer(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    idt: torch.Tensor,
    act: int,
    resnet: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pre = x @ w
    if b.numel():
        pre = pre + b
    y = _act(pre, act)
    if idt.numel():
        y = y * idt
    if resnet:
        din, dout = w.shape
        if dout == din:
            y = y + x
        elif dout == 2 * din:
            y = y + torch.cat([x, x], dim=-1)
    return pre, y


def _cpu_embedding(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    atype: torch.Tensor,
    pair_table: torch.Tensor,
    gate_table: torch.Tensor,
    davg: torch.Tensor,
    dstd: torch.Tensor,
    w1: torch.Tensor,
    idt1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    idt2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    idt3: torch.Tensor,
    act: int,
    type_one_side: int,
    smooth: int,
    ntypes: int,
    resnet2: int,
    resnet3: int,
    rcut: float,
    rcut_smth: float,
    protection: float,
) -> tuple[torch.Tensor, ...]:
    """Environment matrix, embedding MLP and (strip) type-pair gate (fp32).

    Returns ``(rr, pre2, pre3, g, gg)`` with ``gg`` the gated layer-3 output
    that feeds the moment (``gg is g`` in concat mode).
    """
    ev = edge_vec.to(torch.float32)
    src, dst = edge_index[0], edge_index[1]
    center_type, nei_type = atype[dst], atype[src]
    length = ev.norm(dim=-1, keepdim=True)
    length = length + (~edge_mask[:, None]).to(length.dtype)
    q = length + protection
    u = ((length - rcut_smth) / (rcut - rcut_smth)).clamp(0.0, 1.0)
    sw = u**3 * (-6 * u**2 + 15 * u - 10) + 1.0
    em = torch.cat([sw / q, ev * (sw / q**2)], dim=-1)
    rr = (em - davg[center_type]) / dstd[center_type]
    strip = gate_table.shape[0] > 0
    pair_idx = nei_type if type_one_side else center_type * ntypes + nei_type
    pre1 = rr[:, :1] * w1[0] + pair_table[0 if strip else pair_idx]
    h1 = _act(pre1, act)
    if idt1.numel():
        h1 = h1 * idt1
    pre2, h2 = _cpu_layer(h1, w2, b2, idt2, act, resnet2)
    pre3, g = _cpu_layer(h2, w3, b3, idt3, act, resnet3)
    gg = g
    if strip:
        gate = gate_table[pair_idx]
        if smooth:
            gate = gate * sw
        gg = g * (1.0 + gate)
    return rr, pre2, pre3, g, gg


def _cpu_outputs(
    rr: torch.Tensor,
    gg: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    atype: torch.Tensor,
    type_embedding: torch.Tensor | None,
    ng: int,
    axis: int,
    concat_tebd: int,
    nnei: float,
    n_node: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ggm = gg * edge_mask[:, None].to(gg.dtype)
    outer = rr[:, :, None] * ggm[:, None, :]  # (E, 4, ng)
    gr = torch.zeros(n_node, 4, ng, dtype=gg.dtype, device=gg.device)
    gr.index_add_(0, edge_index[1], outer)
    gr = gr / nnei
    gr_t = gr.permute(0, 2, 1)  # (N, ng, 4)
    grrg = torch.matmul(gr_t, gr[:, :, :axis]).reshape(n_node, ng * axis)
    rot_mat = gr_t[:, :, 1:4].contiguous()
    if concat_tebd:
        grrg = torch.cat([grrg, type_embedding[atype]], dim=-1)
    return grrg.contiguous(), rot_mat, gr


def _cpu_forward(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    davg: torch.Tensor,
    dstd: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    idt1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    idt2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    idt3: torch.Tensor,
    gate_table: torch.Tensor,
    act: int,
    type_one_side: int,
    concat_tebd: int,
    write_rotation: int,
    smooth: int,
    axis: int,
    resnet2: int,
    resnet3: int,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
) -> tuple[torch.Tensor, ...]:
    n_edge = edge_vec.shape[0]
    n_node = atype.shape[0]
    n2, ng = w2.shape[1], w3.shape[1]
    ntypes = type_embedding.shape[0]
    strip = gate_table.shape[0] > 0
    pair_table = _cpu_pair_table(type_embedding, w1, b1, type_one_side, strip)
    rr, pre2, pre3, g, gg = _cpu_embedding(
        edge_vec,
        edge_index,
        edge_mask,
        atype,
        pair_table,
        gate_table,
        davg,
        dstd,
        w1,
        idt1,
        w2,
        b2,
        idt2,
        w3,
        b3,
        idt3,
        act,
        type_one_side,
        smooth,
        ntypes,
        resnet2,
        resnet3,
        rcut,
        rcut_smth,
        protection,
    )
    grrg, rot_mat, gr = _cpu_outputs(
        rr,
        gg,
        edge_index,
        edge_mask,
        atype,
        type_embedding,
        ng,
        axis,
        concat_tebd,
        nnei,
        n_node,
    )
    if not write_rotation:
        rot_mat = rot_mat.new_empty(0, ng, 3)
    e_pad = (n_edge + _TILE_EDGES - 1) // _TILE_EDGES * _TILE_EDGES
    dev = edge_vec.device
    pre2_saved = torch.zeros(n2, e_pad, dtype=torch.float32, device=dev)
    pre2_saved[:, :n_edge] = pre2.t()
    g_saved = torch.zeros(ng, e_pad, dtype=torch.float32, device=dev)
    g_saved[:, :n_edge] = (g if act == 0 else pre3).t()
    edge_order = torch.arange(n_edge, dtype=torch.int32, device=dev)
    return grrg, rot_mat, gr, edge_order, pair_table, pre2_saved, g_saved


def _cpu_backward(
    d_grrg: torch.Tensor,
    d_rot_mat: torch.Tensor | None,
    gr: torch.Tensor,
    edge_order: torch.Tensor,
    pair_table: torch.Tensor,
    pre2_saved: torch.Tensor,
    g_saved: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    atype: torch.Tensor,
    davg: torch.Tensor,
    dstd: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    idt1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    idt2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    idt3: torch.Tensor,
    gate_table: torch.Tensor,
    act: int,
    type_one_side: int,
    smooth: int,
    axis: int,
    resnet2: int,
    resnet3: int,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
) -> torch.Tensor:
    n_node = atype.shape[0]
    ng = w3.shape[1]
    strip = gate_table.shape[0] > 0
    n_pairs = gate_table.shape[0] if strip else pair_table.shape[0]
    ntypes = n_pairs if type_one_side else round(n_pairs**0.5)
    ev = edge_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        rr, _pre2, _pre3, _g, gg = _cpu_embedding(
            ev,
            edge_index,
            edge_mask,
            atype,
            pair_table,
            gate_table,
            davg,
            dstd,
            w1,
            idt1,
            w2,
            b2,
            idt2,
            w3,
            b3,
            idt3,
            act,
            type_one_side,
            smooth,
            ntypes,
            resnet2,
            resnet3,
            rcut,
            rcut_smth,
            protection,
        )
        grrg, rot_mat, _gr = _cpu_outputs(
            rr,
            gg,
            edge_index,
            edge_mask,
            atype,
            None,
            ng,
            axis,
            0,
            nnei,
            n_node,
        )
        loss = (grrg * d_grrg[:, : grrg.shape[1]].to(grrg.dtype)).sum()
        if d_rot_mat is not None:
            loss = loss + (rot_mat * d_rot_mat.to(rot_mat.dtype)).sum()
        (d_ev,) = torch.autograd.grad(loss, ev)
    return d_ev.to(edge_vec.dtype)


# ======================================================================
# Registration and the public wrapper
# ======================================================================
_cpu_library: torch.library.Library | None = None


def ensure_registered() -> None:
    """Register the fake / backward / CPU implementations for the ops.

    Idempotent; a no-op when the C++ operator library is not loaded (so the
    plain pt inference path, which never dispatches here, does not require
    the ops).
    """
    global _cpu_library
    if _cpu_library is not None or not op_available():
        return
    torch.library.register_fake("deepmd::dpa1_graph_descriptor")(_forward_fake)
    torch.library.register_fake("deepmd::dpa1_graph_descriptor_backward")(
        _backward_fake
    )
    torch.library.register_autograd(
        "deepmd::dpa1_graph_descriptor", _backward, setup_context=_setup_context
    )
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("dpa1_graph_descriptor", _cpu_forward, "CPU")
    _cpu_library.impl("dpa1_graph_descriptor_backward", _cpu_backward, "CPU")


def _strip_gate_table(se: Any, type_embedding: torch.Tensor) -> torch.Tensor:
    """Type-pair gate table (T or T^2, ng) through the strip embedding net.

    One-side embeds the neighbor-type rows; two-side embeds every
    ``[neighbor, center]`` pair, matching the dense reference's
    ``cal_g_strip`` input layout.
    """
    if se.type_one_side:
        pairs = type_embedding
    else:
        ntypes, tebd_dim = type_embedding.shape
        nei = type_embedding.view(1, ntypes, tebd_dim).expand(ntypes, ntypes, tebd_dim)
        center = type_embedding.view(ntypes, 1, tebd_dim).expand(
            ntypes, ntypes, tebd_dim
        )
        pairs = torch.cat([nei, center], dim=-1).reshape(ntypes * ntypes, 2 * tebd_dim)
    return se.embeddings_strip[0].call(pairs).to(torch.float32).contiguous()


def dpa1_graph_descriptor(
    desc: Any,
    graph: Any,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    write_rotation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the fused DPA1 descriptor from the edge stream.

    Numerically equivalent to the dpmodel reference on the attention-free
    configuration (up to fp32 summation order), in both tebd input modes:
    concat, and strip with or without the smooth type-embedding gate.
    ``edge_vec`` may arrive fp64 (the model-agnostic ``.pt2`` ABI); the cast
    to the fp32 compute happens inside the operator, whose backward returns
    ``d_edge_vec`` in the leaf dtype.

    Parameters
    ----------
    desc : DescrptDPA1
        The pt_expt descriptor module; must satisfy
        ``desc._fused_eligible("cuda")``.
    graph : NeighborGraph
        The lowered neighbor graph: ``edge_vec`` (E, 3) is the autograd leaf
        for the force / virial backward, ``edge_index`` (2, E) the
        ``[src, dst]`` endpoints, ``edge_mask`` (E,) the valid-edge mask.
    atype : torch.Tensor
        Flat node atom types with shape (N,), int64.
    type_embedding : torch.Tensor
        Type embedding table with shape (ntypes + 1, tebd_dim), fp32.
    write_rotation : bool, default: True
        Whether to materialize the equivariant rotation output. Energy-only
        inference can disable it.

    Returns
    -------
    grrg : torch.Tensor
        Descriptor with shape (N, ng * axis [+ tebd_dim]), fp32.
    rot_mat : torch.Tensor
        Equivariant rotation matrix with shape (N, ng, 3), fp32.
    """
    ensure_registered()
    se = desc.se_atten
    layers = se.embeddings[0].layers
    empty = layers[0].w.new_empty(0)

    def optional(t: torch.Tensor | None) -> torch.Tensor:
        return t.contiguous() if t is not None else empty

    if se.tebd_input_mode == "strip":
        gate_table = _strip_gate_table(se, type_embedding)
        smooth = int(se.smooth)
    else:
        gate_table = empty.reshape(0, 0)
        smooth = 0
    w1, w2, w3 = (layer.w.contiguous() for layer in layers)
    grrg, rot_mat, *_aux = torch.ops.deepmd.dpa1_graph_descriptor(
        graph.edge_vec.contiguous(),
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        atype.contiguous(),
        type_embedding.contiguous(),
        # mean / stddev are slot-independent; slot 0 is the canonical (T, 4).
        se.mean[:, 0, :].contiguous(),
        se.stddev[:, 0, :].contiguous(),
        w1,
        optional(layers[0].b),
        optional(layers[0].idt),
        w2,
        optional(layers[1].b),
        optional(layers[1].idt),
        w3,
        optional(layers[2].b),
        optional(layers[2].idt),
        gate_table,
        ACT_CODES[str(layers[0].activation_function).lower()],
        int(se.type_one_side),
        int(desc.concat_output_tebd),
        int(write_rotation),
        smooth,
        int(se.axis_neuron),
        int(layers[1].resnet),
        int(layers[2].resnet),
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
    )
    return grrg, rot_mat
