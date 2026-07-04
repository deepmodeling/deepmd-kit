# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused DPA1 compressed graph-lower descriptor operator.

The CUDA operator ``deepmd::dpa1_graph_compress`` (see
``source/op/pt/dpa1_graph_compress.cu``) evaluates the geo-compressed,
attention-free strip descriptor from the flat edge stream -- environment matrix,
quintic spline table lookup of the geometric embedding, the precomputed strip
type-pair gate, moment reduction and the ``G^T G`` contraction -- in two mega
kernels (forward and backward). This module mirrors
:mod:`.graph_descriptor` for the compressed path:

* **Fake (meta) implementations** for ``make_fx`` / ``torch.export``.
* **A registered backward** routing to ``deepmd::dpa1_graph_compress_backward``.
* **CPU reference implementations** for the freeze pipeline's trace-time
  sample evaluation.

The ``gate_table`` argument carries the precomputed strip type-pair embedding
(:attr:`DescrptDPA1.type_embd_data`); ``compress_data`` / ``compress_info`` hold
the quintic spline coefficients and segment metadata of the tabulated geometric
net.
"""

from typing import (
    Any,
)

import torch

__all__ = [
    "dpa1_graph_compress",
    "dpa1_graph_compress_energy_force",
    "ef_op_available",
    "ensure_registered",
    "mega_eligible",
    "op_available",
]

# Instantiated table widths in dpa1_graph_compress.cu; each divides the 256
# threads per block, so the moment walk splits channels evenly. An arbitrary
# ``ng`` is served by the smallest width that is at least ``ng``, zero-padding
# the spline table and the type-pair gate so the extra channels contribute
# nothing, and slicing them off the descriptor.
_KERNEL_WIDTHS = (8, 16, 32, 64, 128, 256)

_registered = False


def _bucket_width(ng: int) -> int:
    """Smallest instantiated kernel width that holds ``ng`` channels."""
    for w in _KERNEL_WIDTHS:
        if w >= ng:
            return w
    raise ValueError(
        f"dpa1_graph_compress: ng={ng} exceeds the maximum table width "
        f"{_KERNEL_WIDTHS[-1]}"
    )


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa1_graph_compress`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa1_graph_compress", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def ef_op_available() -> bool:
    """Whether all operators required by the fused inference assembly are loaded."""
    return (
        op_available()
        and isinstance(
            getattr(torch.ops.deepmd, "graph_fitting", None),
            torch._ops.OpOverloadPacket,
        )
        and isinstance(
            getattr(torch.ops.deepmd, "edge_force_virial", None),
            torch._ops.OpOverloadPacket,
        )
    )


def mega_eligible(desc: Any) -> bool:
    """Whether ``desc`` can use the fused energy-force mega operator.

    The mega operator feeds the tabulated descriptor straight into the energy
    fitting, so the descriptor width the kernel emits must equal the fitting
    input width. This holds only when ``ng`` is an instantiated kernel width
    (no bucket zero-padding). A non-bucket ``ng`` stays on the level-1 lower,
    where the padding channels are sliced off before the fitting.
    """
    ng = int(desc.se_atten.neuron[-1])
    return _bucket_width(ng) == ng


# ======================================================================
# Fake (meta) implementations
# ======================================================================
def _forward_fake(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    davg: torch.Tensor,
    inverse_stddev: torch.Tensor,
    table: torch.Tensor,
    gate_table: torch.Tensor,
    type_one_side: int,
    concat_tebd: int,
    write_rotation: int,
    smooth: int,
    axis: int,
    canonical: bool,
    lower: float,
    upper: float,
    table_max: float,
    stride0: float,
    stride1: float,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
) -> tuple[torch.Tensor, ...]:
    n_node = atype.shape[0]
    ng = table.shape[1] // 6
    _, tebd_dim = type_embedding.shape
    out_dim = ng * axis + (tebd_dim if concat_tebd else 0)
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
    )


def _backward_fake(
    d_grrg: torch.Tensor,
    d_rot_mat: torch.Tensor | None,
    gr: torch.Tensor,
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
        destination_order,
        destination_row_ptr,
        atype,
        type_embedding,
        davg,
        inverse_stddev,
        table,
        gate_table,
        type_one_side,
        concat_tebd,
        _write_rotation,
        smooth,
        axis,
        canonical,
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        rcut,
        rcut_smth,
        protection,
        nnei,
    ) = inputs
    (gr,) = output[2:]
    ctx.save_for_backward(
        gr,
        edge_vec,
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        atype,
        davg,
        inverse_stddev,
        table,
        gate_table,
    )
    ctx.scalars = (
        type_one_side,
        concat_tebd,
        smooth,
        axis,
        canonical,
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        rcut,
        rcut_smth,
        protection,
        nnei,
    )
    ctx.set_materialize_grads(False)


def _backward(
    ctx: Any,
    d_grrg: torch.Tensor,
    d_rot_mat: torch.Tensor | None,
    *d_aux: Any,
) -> tuple:
    (
        gr,
        edge_vec,
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        atype,
        davg,
        inverse_stddev,
        table,
        gate_table,
    ) = ctx.saved_tensors
    (
        type_one_side,
        concat_tebd,
        smooth,
        axis,
        canonical,
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        rcut,
        rcut_smth,
        protection,
        nnei,
    ) = ctx.scalars
    d_edge_vec = torch.ops.deepmd.dpa1_graph_compress_backward(
        d_grrg,
        d_rot_mat,
        gr,
        edge_vec,
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        atype,
        davg,
        inverse_stddev,
        table,
        gate_table,
        type_one_side,
        smooth,
        axis,
        canonical,
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        rcut,
        rcut_smth,
        protection,
        nnei,
    )
    return (d_edge_vec,) + (None,) * 25


# ======================================================================
# CPU reference implementations
# ======================================================================
def _cpu_tabulate_fusion(
    table: torch.Tensor,
    lower: float,
    upper: float,
    table_max: float,
    stride0: float,
    stride1: float,
    em_x: torch.Tensor,
    last_layer_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Quintic spline table lookup with C1 extrapolation (fp32).

    Parameters
    ----------
    table
        Coefficient table with shape (nspline, last_layer_size * 6).
    lower, upper, table_max, stride0, stride1
        Spline segment bounds and strides (``compress_info`` scalars).
    em_x
        Lookup coordinates with shape (nloc, nnei).
    last_layer_size
        Output width ``ng``.
    reference
        Tensor whose dtype/device define the compute context.

    Returns
    -------
    torch.Tensor
        Tabulated values with shape (nloc, nnei, last_layer_size).
    """
    nloc, nnei = em_x.shape[:2]
    xx = em_x.reshape(nloc, nnei)
    tbl = table.to(device=reference.device, dtype=reference.dtype)

    zeros = torch.zeros(xx.shape, dtype=torch.int64, device=xx.device)
    nspline = tbl.shape[0]
    last_idx = torch.full(xx.shape, nspline - 1, dtype=torch.int64, device=xx.device)
    first_stride = int((upper - lower) / stride0)
    first_stride_value = float(first_stride)

    first_idx = torch.floor((xx - lower) / stride0).to(torch.int64)
    second_idx = first_stride + torch.floor((xx - upper) / stride1).to(torch.int64)
    table_idx = torch.where(
        xx < lower,
        zeros,
        torch.where(
            xx < upper,
            first_idx,
            torch.where(xx < table_max, second_idx, last_idx),
        ),
    )
    table_idx = torch.clamp(table_idx, min=0, max=nspline - 1)

    table_idx_value = table_idx.to(reference.dtype)
    dx_first = xx - (table_idx_value * stride0 + lower)
    dx_second = xx - ((table_idx_value - first_stride_value) * stride1 + upper)
    dx_high = table_max - (
        (last_idx.to(reference.dtype) - first_stride_value) * stride1 + upper
    )
    dx = torch.where(
        xx < lower,
        torch.zeros_like(xx),
        torch.where(
            xx < upper,
            dx_first,
            torch.where(xx < table_max, dx_second, dx_high),
        ),
    )
    extrapolate_delta = torch.where(
        xx < lower,
        xx - lower,
        torch.where(xx >= table_max, xx - table_max, torch.zeros_like(xx)),
    )

    coeff = tbl[table_idx.reshape(-1)]
    coeff = coeff.reshape(nloc, nnei, last_layer_size, 6)
    dx = dx.reshape(nloc, nnei, 1)
    values = (
        coeff[..., 0]
        + (
            coeff[..., 1]
            + (
                coeff[..., 2]
                + (coeff[..., 3] + (coeff[..., 4] + coeff[..., 5] * dx) * dx) * dx
            )
            * dx
        )
        * dx
    )
    values_grad = (
        coeff[..., 1]
        + (
            2 * coeff[..., 2]
            + (3 * coeff[..., 3] + (4 * coeff[..., 4] + 5 * coeff[..., 5] * dx) * dx)
            * dx
        )
        * dx
    )
    extrapolate_delta = extrapolate_delta.reshape(nloc, nnei, 1)
    return values + values_grad * extrapolate_delta


def _cpu_tabulate_se_atten(
    table: torch.Tensor,
    info: tuple[float, float, float, float, float],
    em_x: torch.Tensor,
    em: torch.Tensor,
    two_embed: torch.Tensor,
    last_layer_size: int,
) -> torch.Tensor:
    """Per-edge tabulate_fusion_se_atten body returning the moment outer factor.

    Returns ``(E, 4, ng)`` with ``em`` shaped ``(E, 1, 4)``.
    """
    values = _cpu_tabulate_fusion(table, *info, em_x, last_layer_size, em)
    values = values * two_embed + values
    return (em.unsqueeze(-1) * values.unsqueeze(2)).sum(dim=1)


def _cpu_pair_idx(
    edge_index: torch.Tensor,
    atype: torch.Tensor,
    type_one_side: int,
    ntypes: int,
) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    nei_type, center_type = atype[src], atype[dst]
    if type_one_side:
        return nei_type
    return center_type * ntypes + nei_type


def _cpu_env_and_gg(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    atype: torch.Tensor,
    gate_table: torch.Tensor,
    davg: torch.Tensor,
    dstd: torch.Tensor,
    table: torch.Tensor,
    info: tuple[float, float, float, float, float],
    type_one_side: int,
    smooth: int,
    ntypes: int,
    ng: int,
    rcut: float,
    rcut_smth: float,
    protection: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Environment matrix, tabulated geometric net and strip gate (fp32).

    Returns ``(rr, outer)`` where ``outer`` is ``(E, 4, ng)`` before the
    neighbor-axis reduction.
    """
    ev = edge_vec.to(torch.float32)
    src, dst = edge_index[0], edge_index[1]
    center_type = atype[dst]
    length = ev.norm(dim=-1, keepdim=True)
    length = length + (~edge_mask[:, None]).to(length.dtype)
    q = length + protection
    u = ((length - rcut_smth) / (rcut - rcut_smth)).clamp(0.0, 1.0)
    sw = u**3 * (-6 * u**2 + 15 * u - 10) + 1.0
    em = torch.cat([sw / q, ev * (sw / q**2)], dim=-1)
    rr = (em - davg[center_type]) / dstd[center_type]
    ss = rr[:, 0:1]
    pair_idx = _cpu_pair_idx(edge_index, atype, type_one_side, ntypes)
    gate = gate_table[pair_idx]
    if smooth:
        gate = gate * sw
    n_edge = edge_vec.shape[0]
    em_x = ss.reshape(n_edge, 1)
    em_rr = rr.reshape(n_edge, 1, 4)
    two_embed = gate.reshape(n_edge, 1, ng)
    outer = _cpu_tabulate_se_atten(table, info, em_x, em_rr, two_embed, ng)
    return rr, outer


def _cpu_outputs(
    rr: torch.Tensor,
    outer: torch.Tensor,
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
    outer = outer * edge_mask[:, None, None].to(outer.dtype)
    gr = torch.zeros(n_node, 4, ng, dtype=outer.dtype, device=outer.device)
    gr.index_add_(0, edge_index[1], outer)
    gr = gr / nnei
    gr_t = gr.permute(0, 2, 1)
    grrg = torch.matmul(gr_t, gr[:, :, :axis]).reshape(n_node, ng * axis)
    rot_mat = gr_t[:, :, 1:4].contiguous()
    if concat_tebd:
        grrg = torch.cat([grrg, type_embedding[atype]], dim=-1)
    return grrg.contiguous(), rot_mat, gr


def _cpu_forward(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    davg: torch.Tensor,
    inverse_stddev: torch.Tensor,
    table: torch.Tensor,
    gate_table: torch.Tensor,
    type_one_side: int,
    concat_tebd: int,
    write_rotation: int,
    smooth: int,
    axis: int,
    canonical: bool,
    lower: float,
    upper: float,
    table_max: float,
    stride0: float,
    stride1: float,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
) -> tuple[torch.Tensor, ...]:
    n_node = atype.shape[0]
    ng = table.shape[1] // 6
    ntypes = type_embedding.shape[0]
    if n_node == 0:
        out_dim = ng * axis + (type_embedding.shape[1] if concat_tebd else 0)
        return (
            table.new_empty(0, out_dim),
            table.new_empty(0, ng, 3),
            table.new_empty(0, 4, ng),
        )
    rr, outer = _cpu_env_and_gg(
        edge_vec,
        edge_index,
        edge_mask,
        atype,
        gate_table,
        davg,
        torch.reciprocal(inverse_stddev),
        table,
        (lower, upper, table_max, stride0, stride1),
        type_one_side,
        smooth,
        ntypes,
        ng,
        rcut,
        rcut_smth,
        protection,
    )
    grrg, rot_mat, gr = _cpu_outputs(
        rr,
        outer,
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
    return grrg, rot_mat, gr


def _cpu_backward(
    d_grrg: torch.Tensor,
    d_rot_mat: torch.Tensor | None,
    gr: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    davg: torch.Tensor,
    inverse_stddev: torch.Tensor,
    table: torch.Tensor,
    gate_table: torch.Tensor,
    type_one_side: int,
    smooth: int,
    axis: int,
    canonical: bool,
    lower: float,
    upper: float,
    table_max: float,
    stride0: float,
    stride1: float,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
) -> torch.Tensor:
    n_node = atype.shape[0]
    ng = table.shape[1] // 6
    if n_node == 0:
        return torch.zeros_like(edge_vec)
    ntypes = gate_table.shape[0] if type_one_side else round(gate_table.shape[0] ** 0.5)
    ev = edge_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        rr, outer = _cpu_env_and_gg(
            ev,
            edge_index,
            edge_mask,
            atype,
            gate_table,
            davg,
            torch.reciprocal(inverse_stddev),
            table,
            (lower, upper, table_max, stride0, stride1),
            type_one_side,
            smooth,
            ntypes,
            ng,
            rcut,
            rcut_smth,
            protection,
        )
        grrg, rot_mat, _gr = _cpu_outputs(
            rr,
            outer,
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

    Idempotent; a no-op when the C++ operator library is not loaded.
    """
    global _registered, _cpu_library
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::dpa1_graph_compress")(_forward_fake)
    torch.library.register_fake("deepmd::dpa1_graph_compress_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa1_graph_compress", _backward, setup_context=_setup_context
    )
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("dpa1_graph_compress", _cpu_forward, "CPU")
    _cpu_library.impl("dpa1_graph_compress_backward", _cpu_backward, "CPU")
    _registered = True


def dpa1_graph_compress(
    desc: Any,
    graph: Any,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the fused geo-compressed DPA1 strip descriptor from the edge stream.

    Numerically equivalent to the dense :meth:`DescrptDPA1._call_compressed`
    body on the attention-free strip configuration (up to fp32 summation order).
    ``edge_vec`` may arrive fp64; the operator computes in fp32 and the backward
    returns ``d_edge_vec`` in the leaf dtype.

    Parameters
    ----------
    desc : DescrptDPA1
        The pt_expt descriptor module with ``compress`` and ``geo_compress``
        enabled; must satisfy ``desc._fused_eligible("cuda")``.
    graph : NeighborGraph
        Lowered neighbor graph with ``edge_vec`` (E, 3), ``edge_index`` (2, E)
        and ``edge_mask`` (E,), plus a destination CSR view. The CSR may cover
        cached skin edges whose current cutoff mask is false.
    atype : torch.Tensor
        Flat node atom types with shape (N,), int64.
    type_embedding : torch.Tensor
        Type embedding table with shape (ntypes + 1, tebd_dim), fp32.

    Returns
    -------
    grrg : torch.Tensor
        Descriptor with shape (N, ng * axis [+ tebd_dim]), fp32.
    rot_mat : torch.Tensor
        Equivariant rotation matrix with shape (N, ng, 3), fp32.
    """
    ensure_registered()
    se = desc.se_atten
    ng = int(se.neuron[-1])
    axis = int(se.axis_neuron)
    # The kernel is instantiated at a fixed set of table widths; a descriptor
    # whose ``ng`` is not one of them runs at the next larger width with the
    # spline table and type-pair gate zero-padded, so the extra channels are
    # identically zero and are sliced off below.
    width = _bucket_width(ng)
    pad = width - ng
    compress_data = desc.compress_data[0].contiguous()
    gate_table = desc.type_embd_data.contiguous()
    if pad:
        compress_data = torch.nn.functional.pad(compress_data, (0, pad * 6))
        gate_table = torch.nn.functional.pad(gate_table, (0, pad))
    # The spline segment bounds / strides are baked as scalar constants (not a
    # host tensor) so the exported graph carries no CPU-resident operand. The
    # compress-info buffer is a graph constant; its values are read with proxy
    # tracing disabled so make_fx bakes plain floats rather than proxying the
    # scalar extraction.
    from torch.fx.experimental.proxy_tensor import (
        disable_proxy_modes_tracing,
    )

    with disable_proxy_modes_tracing():
        lower, upper, table_max, stride0, stride1 = (
            float(x) for x in desc.compress_info[0].tolist()[:5]
        )
    if graph.destination_order is None or graph.destination_row_ptr is None:
        raise ValueError("dpa1_graph_compress requires destination CSR topology")
    grrg, rot_mat, _moment = torch.ops.deepmd.dpa1_graph_compress(
        graph.edge_vec.contiguous(),
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        type_embedding.contiguous(),
        se.mean[:, 0, :].contiguous(),
        torch.reciprocal(se.stddev[:, 0, :]).contiguous(),
        compress_data,
        gate_table,
        int(se.type_one_side),
        int(desc.concat_output_tebd),
        1,
        int(se.smooth),
        int(se.axis_neuron),
        bool(graph.destination_sorted),
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
    )
    if pad:
        # Drop the padding channels: the descriptor is stored channel-major
        # (i * axis + j, i in [0, width)), so the real block is the first
        # ng * axis columns; any concatenated type-embedding tail follows the
        # full width * axis span. The rotation matrix keeps its first ng rows.
        desc_block = grrg[:, : ng * axis]
        if desc.concat_output_tebd:
            grrg = torch.cat([desc_block, grrg[:, width * axis :]], dim=-1)
        else:
            grrg = desc_block
        rot_mat = rot_mat[:, :ng, :]
    return grrg, rot_mat


def dpa1_graph_compress_energy_force(
    desc: Any,
    fit: Any,
    graph: Any,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
    node_capacity: int,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate compressed DPA1 energy, force, and virial without autograd.

    The descriptor forward/backward and force assembly remain explicit custom
    operators in the exported graph. The fitting forward/backward is evaluated
    between them, so force is returned as a value and no autograd tape is
    retained. Destination/source CSR tensors are part of the graph contract and
    are reused by both descriptor and force kernels. A graph marked
    ``destination_sorted`` selects direct destination-major addressing; edge
    masks remain authoritative in both addressing modes. This inference-only
    path suppresses the unused rotation output, and fitting backward does not
    retain the descriptor after its last value use.

    Parameters
    ----------
    desc : DescrptDPA1
        The compressed pt_expt descriptor; must satisfy ``mega_eligible(desc)``
        and ``desc._fused_eligible("cuda")``.
    fit : EnergyFittingNet
        The pt_expt fitting module (see
        :func:`~deepmd.kernels.cuda.graph_fitting.fitting_eligible`).
    graph : NeighborGraph
        The lowered neighbor graph (``edge_vec``, ``edge_index``, ``edge_mask``,
        ``n_node``) with destination/source CSR. ``destination_sorted`` must be
        true only when ``destination_order`` is the identity permutation.
    atype : torch.Tensor
        Flat node atom types with shape (N,), int64.
    type_embedding : torch.Tensor
        Type embedding table with shape (ntypes + 1, tebd_dim), fp32.
    ownership : torch.Tensor
        Energy-contributing node mask with shape (N,), bool.
    atom_bias : torch.Tensor
        Combined fitting and atomic-model bias with shape (ntypes,).
    node_capacity : int
        Padded node-axis size ``N`` (the force / atom-virial scatter bound).
    do_atomic_virial : bool
        Whether to also assemble the per-atom virial.

    Returns
    -------
    energy : torch.Tensor
        Per-frame energy with shape (nf, 1), fp64.
    atom_energy : torch.Tensor
        Per-atom energy with shape (N, 1), fp64.
    force : torch.Tensor
        Per-atom force with shape (N, 3), fp32.
    virial : torch.Tensor
        Per-frame virial with shape (nf, 3, 3), fp32.
    atom_virial : torch.Tensor
        Per-atom virial with shape (N, 3, 3) when requested, else empty (0, 3, 3).
    """
    from deepmd.kernels.cuda.edge_force_virial import (
        edge_force_virial,
    )
    from deepmd.kernels.cuda.edge_force_virial import (
        ensure_registered as ensure_force_registered,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        ensure_registered as ensure_fitting_registered,
    )

    ensure_registered()
    ensure_fitting_registered()
    ensure_force_registered()
    if (
        graph.destination_order is None
        or graph.destination_row_ptr is None
        or graph.source_row_ptr is None
        or graph.source_order is None
    ):
        raise ValueError(
            "compressed DPA1 inference requires destination/source CSR topology"
        )
    se = desc.se_atten
    compress_data = desc.compress_data[0].contiguous()
    gate_table = desc.type_embd_data.contiguous()
    edge_vec = graph.edge_vec.to(compress_data.dtype).contiguous()
    # Scalar spline metadata baked as constants (see dpa1_graph_compress), read
    # with proxy tracing disabled so make_fx bakes plain floats.
    from torch.fx.experimental.proxy_tensor import (
        disable_proxy_modes_tracing,
    )

    with disable_proxy_modes_tracing():
        lower, upper, table_max, stride0, stride1 = (
            float(x) for x in desc.compress_info[0].tolist()[:5]
        )
    *hidden, head = fit.nets[0].layers
    fempty = hidden[0].w.new_empty(0)
    inverse_stddev = torch.reciprocal(se.stddev[:, 0, :]).contiguous()
    descriptor, _rotation, moment = torch.ops.deepmd.dpa1_graph_compress(
        edge_vec,
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        type_embedding.contiguous(),
        se.mean[:, 0, :].contiguous(),
        inverse_stddev,
        compress_data,
        gate_table,
        int(se.type_one_side),
        int(desc.concat_output_tebd),
        0,
        int(se.smooth),
        int(se.axis_neuron),
        bool(graph.destination_sorted),
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
    )
    weights = [layer.w.contiguous() for layer in hidden]
    biases = [
        layer.b.contiguous() if layer.b is not None else fempty for layer in hidden
    ]
    timesteps = [
        layer.idt.contiguous() if layer.idt is not None else fempty for layer in hidden
    ]
    residuals = [1 if layer.resnet else 0 for layer in hidden]
    head_weight = head.w.reshape(-1).contiguous()
    head_bias = (
        head.b.reshape(-1).to(torch.float32).contiguous()
        if head.b is not None
        else fempty
    )
    atom_bias = atom_bias.to(torch.float64).contiguous()
    from deepmd.kernels.triton.dpa1.activation import (
        ACT_CODES,
    )

    activation = ACT_CODES[str(hidden[0].activation_function).lower()]
    atom_energy_raw, fitting_saved = torch.ops.deepmd.graph_fitting(
        descriptor,
        atype,
        weights,
        biases,
        timesteps,
        residuals,
        head_weight,
        head_bias,
        atom_bias,
        activation,
    )
    owned = ownership[:, None].to(atom_energy_raw.dtype)
    energy_seed = owned
    atom_energy = atom_energy_raw * owned
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
    )

    frame_index = frame_id_from_n_node(
        graph.n_node,
        n_total=atom_energy.shape[0],
    )
    energy = torch.zeros(
        graph.n_node.shape[0],
        1,
        dtype=atom_energy.dtype,
        device=atom_energy.device,
    ).index_add_(0, frame_index, atom_energy)
    del descriptor
    descriptor_gradient = torch.ops.deepmd.graph_fitting_backward(
        energy_seed,
        fitting_saved,
        weights,
        residuals,
        head_weight,
    )
    del fitting_saved
    edge_gradient = torch.ops.deepmd.dpa1_graph_compress_backward(
        descriptor_gradient,
        None,
        moment,
        edge_vec,
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        atype,
        se.mean[:, 0, :].contiguous(),
        inverse_stddev,
        compress_data,
        gate_table,
        int(se.type_one_side),
        int(se.smooth),
        int(se.axis_neuron),
        bool(graph.destination_sorted),
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
    )
    force, atom_virial, virial = edge_force_virial(
        edge_gradient,
        edge_vec,
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        graph.n_node,
        node_capacity,
        do_atomic_virial,
    )
    return energy, atom_energy, force, virial, atom_virial
