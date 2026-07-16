# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the end-to-end DPA1 graph-lower energy-force operator.

The CUDA operator ``deepmd::dpa1_graph_energy_force`` (see
``source/op/pt/dpa1_graph_energy_force.cu``) fuses the whole DPA1 (``se_atten``,
attention-free) energy path -- descriptor, energy fitting, and the analytic
force / virial assembly -- into one graph node. It drives the descriptor and
fitting mega kernels and their backwards directly, computing the force from the
reduced energy internally (``dE_redu/d(atom_energy) == 1``), so no autograd tape
is built and the force is returned as a value.

It is numerically identical to the separate-operator path (descriptor + fitting
forwards with the force from ``autograd.grad``): the same operator backwards run
the same fp32 arithmetic in the same order. The fusion removes the autograd
machinery, the output-agnostic per-component grad loop, and the inter-operator
array glue. It is selected at freeze time by ``DP_CUDA_INFER >= 2``; the
separate-operator level-1 path is unchanged.

The operator is forward-only (the force is an output, not a gradient), so only
the fake and CPU implementations are registered -- no autograd bridge.
"""

from typing import (
    Any,
)

import torch

from deepmd.kernels.cuda.dpa1.graph_descriptor import (
    _strip_gate_table,
)
from deepmd.kernels.cuda.dpa1.graph_descriptor import (
    ensure_registered as ensure_descriptor_registered,
)
from deepmd.kernels.cuda.edge_force_virial import (
    ensure_registered as ensure_force_registered,
)
from deepmd.kernels.cuda.graph_fitting import (
    ensure_registered as ensure_fitting_registered,
)
from deepmd.kernels.triton.dpa1.activation import (
    ACT_CODES,
)

__all__ = [
    "dpa1_graph_energy_force",
    "ensure_registered",
    "op_available",
]


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa1_graph_energy_force`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa1_graph_energy_force", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def _fake(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    source_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    n_node: torch.Tensor,
    ownership: torch.Tensor,
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
    smooth: int,
    axis: int,
    resnet2: int,
    resnet3: int,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
    fit_ws: list[torch.Tensor],
    fit_bs: list[torch.Tensor],
    fit_idts: list[torch.Tensor],
    fit_resnets: list[int],
    w_head: torch.Tensor,
    b_head: torch.Tensor,
    bias_atom_e: torch.Tensor,
    fit_act: int,
    node_capacity: int,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, ...]:
    nf = n_node.shape[0]
    n = atype.shape[0]
    dev = edge_vec.device
    # Force / virial assemble in the model compute precision (descriptor weight
    # dtype), matching the real operator; the energy reduction stays fp64.
    fprec = w1.dtype
    return (
        torch.empty(nf, 1, dtype=torch.float64, device=dev),
        torch.empty(n, 1, dtype=torch.float64, device=dev),
        torch.empty(n, 3, dtype=fprec, device=dev),
        torch.empty(nf, 3, 3, dtype=fprec, device=dev),
        torch.empty(n if do_atomic_virial else 0, 3, 3, dtype=fprec, device=dev),
    )


def _cpu(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    source_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    n_node: torch.Tensor,
    ownership: torch.Tensor,
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
    smooth: int,
    axis: int,
    resnet2: int,
    resnet3: int,
    rcut: float,
    rcut_smth: float,
    protection: float,
    nnei: float,
    fit_ws: list[torch.Tensor],
    fit_bs: list[torch.Tensor],
    fit_idts: list[torch.Tensor],
    fit_resnets: list[int],
    w_head: torch.Tensor,
    b_head: torch.Tensor,
    bias_atom_e: torch.Tensor,
    fit_act: int,
    node_capacity: int,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, ...]:
    """Trace-time reference: the same sequence over the sub-operators' CPU
    implementations, so the exported graph's sample evaluation is faithful.
    """
    grrg, _rot, gr, order, pair_table, pre2_saved, g_saved = (
        torch.ops.deepmd.dpa1_graph_descriptor(
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
            0,
            smooth,
            axis,
            resnet2,
            resnet3,
            rcut,
            rcut_smth,
            protection,
            nnei,
        )
    )
    atom_e_raw, fit_saved = torch.ops.deepmd.graph_fitting(
        grrg,
        atype,
        fit_ws,
        fit_bs,
        fit_idts,
        fit_resnets,
        w_head,
        b_head,
        bias_atom_e,
        fit_act,
    )
    owned = ownership[:, None].to(atom_e_raw.dtype)
    energy_seed = owned
    atom_e = atom_e_raw * owned
    nf = n_node.shape[0]
    frame_id = torch.repeat_interleave(
        torch.arange(nf, dtype=torch.long, device=n_node.device), n_node
    )
    energy = torch.zeros(nf, 1, dtype=atom_e.dtype, device=atom_e.device)
    energy = energy.index_add(0, frame_id, atom_e)
    d_grrg = torch.ops.deepmd.graph_fitting_backward(
        energy_seed, fit_saved, fit_ws, fit_resnets, w_head
    )
    del grrg, fit_saved
    g_e = torch.ops.deepmd.dpa1_graph_descriptor_backward(
        d_grrg,
        None,
        gr,
        order,
        pair_table,
        pre2_saved,
        g_saved,
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
    # Assemble force / virial in the descriptor weight precision, matching
    # ``_fake`` and the CUDA operator; the sub-operator would otherwise return
    # fp64 whenever the edge inputs are fp64.
    fprec = w1.dtype
    force, atom_virial, virial = torch.ops.deepmd.edge_force_virial(
        g_e.to(fprec),
        edge_vec.to(fprec),
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
        n_node,
        node_capacity,
        do_atomic_virial,
    )
    return energy, atom_e, force, virial, atom_virial


_cpu_library: torch.library.Library | None = None


def ensure_registered() -> None:
    """Register the fake and CPU implementations for the op.

    Idempotent; a no-op when the C++ operator library is not loaded. The
    sub-operators must be registered too, since the CPU reference dispatches
    through them.
    """
    global _cpu_library
    if _cpu_library is not None or not op_available():
        return
    ensure_descriptor_registered()
    ensure_fitting_registered()
    ensure_force_registered()
    torch.library.register_fake("deepmd::dpa1_graph_energy_force")(_fake)
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("dpa1_graph_energy_force", _cpu, "CPU")


def dpa1_graph_energy_force(
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
    """Fused energy / force / virial from the edge stream.

    Parameters
    ----------
    desc : DescrptDPA1
        The pt_expt descriptor module; must satisfy ``desc._fused_eligible("cuda")``.
    fit : EnergyFittingNet
        The pt_expt fitting module; must satisfy
        :func:`~deepmd.kernels.cuda.graph_fitting.fitting_eligible`.
    graph : NeighborGraph
        The lowered neighbor graph (``edge_vec``, ``edge_index``, ``edge_mask``,
        ``n_node``) with destination/source CSR permutations.
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
        Per-atom force with shape (N, 3), ``edge_vec`` dtype.
    virial : torch.Tensor
        Per-frame virial with shape (nf, 3, 3), ``edge_vec`` dtype.
    atom_virial : torch.Tensor
        Per-atom virial with shape (N, 3, 3) when requested, else empty (0, 3, 3).
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

    *hidden, head = fit.nets[0].layers
    fempty = hidden[0].w.new_empty(0)
    if (
        graph.destination_order is None
        or graph.destination_row_ptr is None
        or graph.source_order is None
        or graph.source_row_ptr is None
    ):
        raise ValueError(
            "DPA1 fused inference requires destination/source CSR topology"
        )
    return torch.ops.deepmd.dpa1_graph_energy_force(
        graph.edge_vec.contiguous(),
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        graph.source_order.contiguous(),
        graph.source_row_ptr.contiguous(),
        atype.contiguous(),
        graph.n_node,
        ownership.contiguous(),
        type_embedding.contiguous(),
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
        smooth,
        int(se.axis_neuron),
        int(layers[1].resnet),
        int(layers[2].resnet),
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
        [layer.w.contiguous() for layer in hidden],
        [layer.b.contiguous() if layer.b is not None else fempty for layer in hidden],
        [
            layer.idt.contiguous() if layer.idt is not None else fempty
            for layer in hidden
        ],
        [1 if layer.resnet else 0 for layer in hidden],
        head.w.reshape(-1).contiguous(),
        (
            head.b.reshape(-1).to(torch.float32).contiguous()
            if head.b is not None
            else fempty
        ),
        atom_bias.to(torch.float64).contiguous(),
        ACT_CODES[str(hidden[0].activation_function).lower()],
        node_capacity,
        do_atomic_virial,
    )
