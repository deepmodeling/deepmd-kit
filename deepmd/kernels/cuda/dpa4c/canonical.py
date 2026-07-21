# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact canonical deployment path for compressed DPA4C."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

if TYPE_CHECKING:
    from deepmd.pt_expt.utils.canonical_graph import (
        CanonicalGraph,
    )

_cpu_library: torch.library.Library | None = None


def canonical_model_eligible(model: Any) -> bool:
    """Return whether a model can use the compact source-only graph ABI."""
    atomic_model = getattr(model, "atomic_model", None)
    descriptor = getattr(atomic_model, "descriptor", None)
    fitting = getattr(atomic_model, "fitting_net", None)
    if descriptor is None or fitting is None:
        return False
    from deepmd.pt_expt.descriptor.dpa4c import (
        DescrptDPA4C,
    )

    if not isinstance(descriptor, DescrptDPA4C):
        return False
    if not bool(getattr(descriptor, "compress", False)):
        return False
    if getattr(descriptor, "exclude_types", None):
        return False
    if getattr(atomic_model, "pair_excl", None) is not None:
        return False
    if getattr(atomic_model, "atom_excl", None) is not None:
        return False
    from deepmd.kernels.cuda.dpa4c.graph_compress import (
        mega_eligible,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        fitting_eligible,
    )

    return mega_eligible(descriptor) and fitting_eligible(fitting)


def op_available() -> bool:
    """Return whether both compact DPA4C descriptor operators are loaded."""
    forward = getattr(torch.ops.deepmd, "dpa4c_canonical_compress", None)
    backward = getattr(
        torch.ops.deepmd,
        "dpa4c_canonical_compress_backward",
        None,
    )
    backward_inplace = getattr(
        torch.ops.deepmd,
        "dpa4c_canonical_compress_backward_inplace",
        None,
    )
    return all(
        isinstance(operator, torch._ops.OpOverloadPacket)
        for operator in (forward, backward, backward_inplace)
    )


def _forward_fake(
    edge_vec: torch.Tensor,
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    table: torch.Tensor,
    pair_film: torch.Tensor,
    pair_mixing: torch.Tensor,
    type_embedding: torch.Tensor,
    readout_matrices: torch.Tensor,
    coupling_meta: torch.Tensor,
    coupling_entry: torch.Tensor,
    coupling_value: torch.Tensor,
    output_mean: torch.Tensor,
    output_inv_std: torch.Tensor,
    lmax: int,
    table_stride: float,
    table_max: float,
    rcut: float,
    eps: float,
    degree_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        source,
        destination_row_ptr,
        table,
        pair_film,
        pair_mixing,
        readout_matrices,
        coupling_meta,
        coupling_entry,
        coupling_value,
        output_mean,
        output_inv_std,
        table_stride,
        table_max,
        rcut,
        eps,
        degree_floor,
    )
    from deepmd.kernels.cuda.dpa4c.graph_compress import (
        descriptor_profile,
    )

    profile = descriptor_profile(int(type_embedding.shape[1]), int(lmax))
    nodes = atype.shape[0]
    descriptor = edge_vec.new_empty(nodes, profile.output_width, dtype=torch.float32)
    state = edge_vec.new_empty(nodes, profile.state_width, dtype=torch.float32)
    return descriptor, state


def _backward_fake(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    del descriptor_gradient, state, args
    return torch.empty_like(edge_vec)


def _backward_inplace_fake(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    del descriptor_gradient, state, args
    return torch.empty_like(edge_vec)


def _energy_gradient_fake(
    edge_vec: torch.Tensor,
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    del source, destination_row_ptr, args
    return (
        edge_vec.new_empty(atype.shape[0], 1, dtype=torch.float64),
        torch.empty_like(edge_vec),
    )


def _cpu_energy_gradient(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sequence of the fused operator, evaluated in one run."""
    from deepmd.kernels.cuda.graph_fitting import _cpu_backward as fitting_backward
    from deepmd.kernels.cuda.graph_fitting import _cpu_forward as fitting_forward

    descriptor_args = args[:_DESCRIPTOR_ARGUMENT_COUNT]
    ws, bs, resnets, w_head, b_head, bias_atom_e, act, seed, _tile = args[
        _DESCRIPTOR_ARGUMENT_COUNT:
    ]
    descriptor, state = _cpu_forward(*descriptor_args)
    atype = descriptor_args[3]
    energy, saved = fitting_forward(
        descriptor, atype, ws, bs, resnets, w_head, b_head, bias_atom_e, act
    )
    gradient = fitting_backward(
        seed.reshape(-1, 1), saved, ws, bs, resnets, w_head, act
    )
    edge_gradient = _cpu_backward(gradient, state, *descriptor_args)
    return energy, edge_gradient


def _generic_topology(
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize the generic topology for the small CPU trace sample."""
    physical_edge_count = int(destination_row_ptr[-1].item())
    node_count = destination_row_ptr.shape[0] - 1
    destination = torch.repeat_interleave(
        torch.arange(node_count, dtype=torch.int64, device=source.device),
        destination_row_ptr[1:] - destination_row_ptr[:-1],
        output_size=physical_edge_count,
    )
    source_i64 = source.to(torch.int64)
    destination_storage = torch.zeros_like(source_i64)
    destination_storage[:physical_edge_count] = destination
    edge_index = torch.stack((source_i64, destination_storage))
    edge_mask = (
        torch.arange(source.shape[0], dtype=torch.int64, device=source.device)
        < physical_edge_count
    )
    destination_order = torch.arange(
        source.shape[0],
        dtype=torch.int64,
        device=source.device,
    )
    return edge_index, edge_mask, destination_order


# The compact ABI drops the three topology tensors of the generic ABI and its
# leading ``canonical`` flag; the remaining trailing scalars are identical.
_CANONICAL_SCALAR_COUNT = 6

#: Leading arguments of the fused operator that describe the descriptor:
#: ``edge_vec`` plus the compact topology, the compression artifacts and the
#: six trailing geometry scalars.
_DESCRIPTOR_ARGUMENT_COUNT = 20


def _cpu_forward(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from deepmd.kernels.cuda.dpa4c.graph_compress import _cpu_forward as generic_forward

    edge_vec, source, destination_row_ptr, atype, *tail = args
    edge_index, edge_mask, destination_order = _generic_topology(
        source,
        destination_row_ptr,
    )
    return generic_forward(
        edge_vec,
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        atype,
        *tail[:-_CANONICAL_SCALAR_COUNT],
        True,
        *tail[-_CANONICAL_SCALAR_COUNT:],
    )


def _cpu_backward(*args: Any) -> torch.Tensor:
    from deepmd.kernels.cuda.dpa4c.graph_compress import (
        _cpu_backward as generic_backward,
    )

    descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype, *tail = (
        args
    )
    edge_index, edge_mask, destination_order = _generic_topology(
        source,
        destination_row_ptr,
    )
    return generic_backward(
        descriptor_gradient,
        state,
        edge_vec,
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        atype,
        *tail[:-_CANONICAL_SCALAR_COUNT],
        True,
        *tail[-_CANONICAL_SCALAR_COUNT:],
    )


def _cpu_backward_inplace(*args: Any) -> torch.Tensor:
    edge_gradient = _cpu_backward(*args)
    args[1].zero_()
    return edge_gradient


def ensure_registered() -> None:
    """Register fake and CPU implementations for compact DPA4C operators."""
    global _cpu_library
    if _cpu_library is not None or not op_available():
        return
    torch.library.register_fake("deepmd::dpa4c_canonical_compress")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4c_canonical_compress_backward")(
        _backward_fake
    )
    torch.library.register_fake("deepmd::dpa4c_canonical_compress_energy_gradient")(
        _energy_gradient_fake
    )
    torch.library.register_fake("deepmd::dpa4c_canonical_compress_backward_inplace")(
        _backward_inplace_fake
    )
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("dpa4c_canonical_compress", _cpu_forward, "CPU")
    _cpu_library.impl(
        "dpa4c_canonical_compress_backward",
        _cpu_backward,
        "CPU",
    )
    _cpu_library.impl(
        "dpa4c_canonical_compress_energy_gradient",
        _cpu_energy_gradient,
        "CPU",
    )
    _cpu_library.impl(
        "dpa4c_canonical_compress_backward_inplace",
        _cpu_backward_inplace,
        "CPU",
    )


def dpa4c_canonical_compress_energy_force(
    descriptor: Any,
    fitting: Any,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate compressed DPA4C from a compact canonical edge stream.

    The compact ABI carries only source indices and CSR row pointers, so the
    descriptor operator addresses edges by position and omits every mask,
    padding-type, and table-tail check that the cutoff-compacted graph already
    guarantees. The backward reuses the forward state in place.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.
    fitting
        Eligible pt_expt energy fitting network.
    graph
        Canonical graph whose ``source`` has shape ``(E,)`` in ``uint32`` and
        whose ``destination_row_ptr`` has shape ``(N + 1,)`` in ``int64``.
    atype
        Flat node atom types with shape ``(N,)``.
    ownership
        Boolean mask selecting energy-contributing nodes with shape ``(N,)``.
    atom_bias
        Combined atomic energy bias with shape ``(ntypes,)`` in eV.
    do_atomic_virial
        Whether to return per-node virials.

    Returns
    -------
    energy
        Per-frame energy with shape ``(F, 1)`` in eV, fp64.
    atom_energy
        Per-node energy with shape ``(N, 1)`` in eV, fp64.
    force
        Per-node force with shape ``(N, 3)`` in eV/Å, fp32.
    virial
        Per-frame virial with shape ``(F, 3, 3)`` in eV, fp32.
    atom_virial
        Per-node virial with shape ``(N, 3, 3)`` in eV, or an empty tensor.

    Raises
    ------
    ValueError
        If the model or the compiled operators do not support the compact path.
    """
    from deepmd.kernels.cuda.dpa4c.graph_compress import (
        compressed_operator_arguments,
        mega_eligible,
    )
    from deepmd.kernels.cuda.edge_force_virial import (
        canonical_edge_force_virial,
        canonical_op_available,
    )
    from deepmd.kernels.cuda.edge_force_virial import (
        ensure_registered as ensure_force_registered,
    )
    from deepmd.kernels.cuda.edge_force_virial import (
        frame_scalar_sum,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        ensure_registered as ensure_fitting_registered,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        fitting_operator_arguments,
        node_tile,
    )

    ensure_registered()
    ensure_fitting_registered()
    ensure_force_registered()
    if not mega_eligible(descriptor) or not canonical_op_available():
        raise ValueError("model is not eligible for compact canonical DPA4C inference")

    network = fitting_operator_arguments(fitting)
    atom_energy_raw, edge_gradient = (
        torch.ops.deepmd.dpa4c_canonical_compress_energy_gradient(
            graph.edge_vec,
            graph.source,
            graph.destination_row_ptr,
            atype,
            *compressed_operator_arguments(descriptor),
            int(descriptor.lmax),
            *descriptor._compression_scalars,
            network.weights,
            network.biases,
            network.residuals,
            network.head_weight,
            network.head_bias,
            atom_bias.to(torch.float64).contiguous(),
            network.activation,
            ownership.to(torch.float64).reshape(-1).contiguous(),
            node_tile(),
        )
    )
    atom_energy = atom_energy_raw * ownership[:, None].to(atom_energy_raw.dtype)
    energy = frame_scalar_sum(atom_energy, graph.n_node)
    force, atom_virial, virial = canonical_edge_force_virial(
        edge_gradient,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        graph.n_node,
        atype.shape[0],
        do_atomic_virial,
    )
    return energy, atom_energy, force, virial, atom_virial
