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

from deepmd.pt_expt.kernels.utils import (
    backend_device_type,
    operator_available,
)

if TYPE_CHECKING:
    from deepmd.pt_expt.utils.canonical_graph import (
        CanonicalGraph,
    )

_cpu_library: torch.library.Library | None = None


def canonical_model_eligible(model: Any) -> bool:
    """Return whether a model can use the compact source-only graph ABI.

    The compact ABI exists so that a device-resident neighbor list can reach
    the descriptor without a host round trip, which is a property of the
    CUDA / Kokkos deployment. Only that backend implements its operators, so
    a CPU target keeps the generic graph lower, whose operators it does
    implement and whose extra cost is one index tensor.
    """
    if backend_device_type() != "cuda":
        return False
    atomic_model = getattr(model, "atomic_model", None)
    # A bridged composition deploys like its learned part: the fused operator
    # evaluates the analytical pair potential in the same edge scan.
    parts = None if atomic_model is None else atomic_model.fused_decomposition()
    if parts is None:
        return False
    descriptor = parts[0].descriptor
    fitting = parts[0].fitting_net
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
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        mega_eligible,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_eligible,
    )

    return mega_eligible(descriptor) and fitting_eligible(fitting)


def op_available() -> bool:
    """Return whether the backend device carries the compact DPA4C suite."""
    return all(
        operator_available(name)
        for name in (
            "dpa4c_canonical_compress",
            "dpa4c_canonical_compress_backward",
            "dpa4c_canonical_compress_backward_inplace",
            "dpa4c_canonical_compress_energy_force",
        )
    )


#: Position of the per-node magnetic moment among the compression arguments,
#: which open with the radial table (see ``compressed_operator_arguments``),
#: and among the inputs that follow ``edge_vec`` in the compact descriptor
#: operators, where the source indices, the destination row pointers and the
#: node types precede those arguments.
_SPIN_ARGUMENT_SLOT = 12
_SPIN_SLOT = 3 + _SPIN_ARGUMENT_SLOT


def _forward_fake(
    edge_vec: torch.Tensor,
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    table: torch.Tensor,
    basis_table: torch.Tensor,
    basis_matrix: torch.Tensor,
    pair_film: torch.Tensor,
    pair_mixing: torch.Tensor,
    type_embedding: torch.Tensor,
    readout_matrices: torch.Tensor,
    coupling_meta: torch.Tensor,
    coupling_entry: torch.Tensor,
    coupling_value: torch.Tensor,
    output_mean: torch.Tensor,
    output_inv_std: torch.Tensor,
    spin: torch.Tensor,
    spin_pair: torch.Tensor,
    spin_type: torch.Tensor,
    contact_radius: torch.Tensor,
    lmax: int,
    table_stride: float,
    table_max: float,
    rcut: float,
    eps: float,
    degree_floor: float,
    f_inner: float,
    f_outer: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        source,
        destination_row_ptr,
        table,
        basis_table,
        basis_matrix,
        pair_film,
        pair_mixing,
        readout_matrices,
        coupling_meta,
        coupling_entry,
        coupling_value,
        output_mean,
        output_inv_std,
        spin_pair,
        spin_type,
        contact_radius,
        table_stride,
        table_max,
        rcut,
        eps,
        degree_floor,
        f_inner,
        f_outer,
    )
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        descriptor_profile,
    )

    profile = descriptor_profile(
        int(type_embedding.shape[1]), int(lmax), spin.ndim == 2
    )
    nodes = atype.shape[0]
    descriptor = edge_vec.new_empty(nodes, profile.output_width, dtype=torch.float32)
    state = edge_vec.new_empty(nodes, profile.state_width, dtype=torch.float32)
    return descriptor, state


def _backward_fake(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del descriptor_gradient, state
    return _backward_shapes(edge_vec, args[_SPIN_SLOT])


def _backward_inplace_fake(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del descriptor_gradient, state
    return _backward_shapes(edge_vec, args[_SPIN_SLOT])


def _backward_shapes(
    edge_vec: torch.Tensor,
    spin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the three backward outputs of one compact call, unpopulated.

    Each absent output is allocated separately, because the schema declares
    three unannotated results and two of them may not share storage.
    """
    if spin.ndim != 2:
        return (
            torch.empty_like(edge_vec),
            edge_vec.new_empty((0,), dtype=torch.float32),
            edge_vec.new_empty((0,), dtype=torch.float32),
        )
    return (
        torch.empty_like(edge_vec),
        spin.new_empty((spin.shape[0], 3)),
        torch.empty_like(edge_vec),
    )


def _energy_force_fake(
    edge_vec: torch.Tensor,
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    n_node: torch.Tensor,
    n_local: torch.Tensor,
    atype: torch.Tensor,
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    del source, destination_row_ptr, source_row_ptr, source_order, n_local
    spin = args[_SPIN_ARGUMENT_SLOT]
    nodes = atype.shape[0]
    frames = n_node.shape[0]
    has_spin = spin.ndim == 2
    # Each output is allocated separately, because the schema declares seven
    # unannotated results and no two of them may share storage.
    return (
        edge_vec.new_empty(frames, 1, dtype=torch.float64),
        edge_vec.new_empty(nodes, 1, dtype=torch.float64),
        edge_vec.new_empty(nodes, 3, dtype=torch.float64),
        edge_vec.new_empty(frames, 3, 3, dtype=torch.float64),
        edge_vec.new_empty(nodes, 3, 3, dtype=torch.float32),
        edge_vec.new_empty(nodes, 3, dtype=torch.float64)
        if has_spin
        else edge_vec.new_empty((0,), dtype=torch.float32),
        edge_vec.new_empty(nodes, dtype=torch.int32),
    )


def _cpu_energy_force(
    edge_vec: torch.Tensor,
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    n_node: torch.Tensor,
    n_local: torch.Tensor,
    atype: torch.Tensor,
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    """Reference sequence of the fused operator, evaluated in one run.

    The force and the virial are assembled by index reduction in fp64. The
    per-atom virial is attributed to the source nodes that the source CSR
    lists, and is zero when two-entry placeholders take the place of the CSR.
    A frame whose energy or virial is not finite reports every force of its
    nodes, and every magnetic force, as not a number, as the CUDA operator
    does.
    """
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
        node_ownership_mask,
    )

    node_count = atype.shape[0]
    edge_storage = edge_vec.shape[0]
    device = edge_vec.device
    complete = (
        source_row_ptr.numel() == node_count + 1
        and source_order.numel() == edge_storage
    )
    if not complete and not (source_row_ptr.numel() == 2 and source_order.numel() == 2):
        raise RuntimeError(
            "dpa4c_canonical_compress_energy_force: the source CSR must hold "
            f"{node_count + 1} row pointers and {edge_storage} positions, or two "
            "entries each to decline the per-atom virial; got "
            f"{source_row_ptr.numel()} and {source_order.numel()}"
        )
    if node_count == 0:
        # An empty node axis holds no energy, force or virial: the outputs
        # take the layout of the operator with zero frame totals, as in the
        # CUDA operator.
        outputs = _energy_force_fake(
            edge_vec,
            source,
            destination_row_ptr,
            source_row_ptr,
            source_order,
            n_node,
            n_local,
            atype,
            *args,
        )
        outputs[0].zero_()
        outputs[3].zero_()
        return outputs
    real = atype >= 0
    mask = real & node_ownership_mask(n_node, n_local, node_count)
    atype = torch.where(real, atype, torch.zeros_like(atype))
    seed = mask.to(torch.float64)
    descriptor_args = (
        edge_vec,
        source,
        destination_row_ptr,
        atype,
        *args[:_DESCRIPTOR_TAIL_COUNT],
    )
    ws, bs, resnets, w_head, b_head, bias_atom_e, act, _tile, pair_table = args[
        _DESCRIPTOR_TAIL_COUNT:
    ]
    descriptor, state = _cpu_forward(*descriptor_args)
    energy, saved = torch.ops.deepmd.graph_fitting(
        descriptor, atype, ws, bs, resnets, w_head, b_head, bias_atom_e, act
    )
    gradient = torch.ops.deepmd.graph_fitting_backward(
        seed.reshape(-1, 1), saved, ws, bs, resnets, w_head, act
    )
    edge_gradient, spin_gradient, edge_spin_gradient, pair_energy = _generic_backward(
        gradient, state, *descriptor_args, pair=(pair_table, seed)
    )
    if pair_table.numel():
        energy = energy + pair_energy
    atom_energy = energy * seed[:, None]

    edge_index, edge_mask, _ = _generic_topology(source, destination_row_ptr)
    keep = edge_mask[:, None].to(torch.float64)
    g = edge_gradient.to(torch.float64) * keep
    outer = -(g[:, :, None] * edge_vec.to(torch.float64)[:, None, :])
    source_index, destination_index = edge_index[0], edge_index[1]
    destination_force = torch.zeros(
        node_count, 3, dtype=torch.float64, device=device
    ).index_add_(0, destination_index, g)
    force = destination_force.clone().index_add_(0, source_index, -g)
    # A force contribution that lies outside the range of the fixed-point totals
    # of the CUDA operator -- an edge onto its source, the row of a destination,
    # or the magnetic contribution of an edge onto its source -- makes its node
    # report its energy as not a number, as the CUDA operator does.
    out_of_range = (destination_force.abs() >= _FORCE_FIXED_LIMIT).any(1)
    out_of_range[source_index[(g.abs() >= _FORCE_FIXED_LIMIT).any(1)]] = True
    if spin_gradient.ndim == 2:
        spin_edges = (edge_spin_gradient.abs() * keep >= _FORCE_FIXED_LIMIT).any(1)
        out_of_range[source_index[spin_edges]] = True
    atom_energy[out_of_range] = torch.nan
    node_virial = torch.zeros(
        node_count, 3, 3, dtype=torch.float64, device=device
    ).index_add_(0, destination_index, outer)
    # Padding nodes past the last frame belong to none: they enter no frame
    # total, as in the CUDA operator.
    frame_id = frame_id_from_n_node(n_node, n_total=node_count)
    in_frame = torch.arange(node_count, dtype=torch.int64, device=device) < n_node.sum()
    virial = torch.zeros(
        n_node.shape[0], 3, 3, dtype=torch.float64, device=device
    ).index_add_(0, frame_id[in_frame], node_virial[in_frame])
    frame_energy = torch.zeros(
        n_node.shape[0], 1, dtype=torch.float64, device=device
    ).index_add_(0, frame_id[in_frame], atom_energy[in_frame])
    atom_virial = torch.zeros(node_count, 3, 3, dtype=torch.float64, device=device)
    if complete:
        listed = source_row_ptr[1:] - source_row_ptr[:-1]
        rows = torch.repeat_interleave(
            torch.arange(node_count, dtype=torch.int64, device=device), listed
        )
        positions = source_order[: int(source_row_ptr[-1])].to(torch.int64)
        atom_virial.index_add_(0, rows, outer[positions])
    if spin_gradient.ndim == 2:
        spin_force = torch.zeros(
            node_count, 3, dtype=torch.float64, device=device
        ).index_add_(0, source_index, edge_spin_gradient.to(torch.float64) * keep)
        force_mag = -(spin_gradient.to(torch.float64) + spin_force)
    else:
        force_mag = spin_gradient.new_empty((0,), dtype=torch.float32)
    # A fixed-point force cannot carry a non-finite value through the CUDA
    # operator, which marks the forces of a broken frame instead; padding
    # nodes past the last frame belong to none.
    broken = ~(
        torch.isfinite(frame_energy).all(1) & torch.isfinite(virial).flatten(1).all(1)
    )
    node_broken = broken[frame_id] & in_frame
    force[node_broken] = torch.nan
    if force_mag.ndim == 2:
        force_mag[node_broken] = torch.nan
    return (
        frame_energy,
        atom_energy,
        force,
        virial,
        atom_virial.to(torch.float32),
        force_mag,
        mask.to(torch.int32),
    )


def _generic_topology(
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize the generic topology of a compact graph for the CPU paths."""
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
_CANONICAL_SCALAR_COUNT = 8

#: Arguments of the compact descriptor operator: ``edge_vec``, the source
#: indices, the destination row pointers and the node types, then the
#: compression artifacts and the trailing geometry scalars.
_DESCRIPTOR_ARGUMENT_COUNT = 28

#: Magnitude in eV/Å from which a force contribution lies outside the range of
#: the fixed-point force totals of the CUDA operator (``kForceFixedLimit``).
_FORCE_FIXED_LIMIT = 2.0**30

#: Descriptor arguments that follow the node types. The fused energy-force
#: operator passes them after its own graph inputs, which insert the source
#: CSR and the frame counts before the node types.
_DESCRIPTOR_TAIL_COUNT = _DESCRIPTOR_ARGUMENT_COUNT - 4


def _cpu_forward(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        _reference_forward as generic_forward,
    )

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


def _generic_backward(
    *args: Any,
    pair: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the generic reference backward on a compact payload.

    ``pair`` is the series table and the energy cotangent of an analytical
    pair potential; the compact descriptor operators carry none.
    """
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        _reference_backward as generic_backward,
    )

    descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype, *tail = (
        args
    )
    edge_index, edge_mask, destination_order = _generic_topology(
        source,
        destination_row_ptr,
    )
    no_pair = edge_vec.new_empty(0)
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
        *((no_pair, no_pair) if pair is None else pair),
    )


def _cpu_backward(*args: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _generic_backward(*args)[:3]


def _cpu_backward_inplace(
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gradients = _cpu_backward(*args)
    args[1].zero_()
    return gradients


def ensure_registered() -> None:
    """Register fake and CPU implementations for compact DPA4C operators."""
    global _cpu_library
    if _cpu_library is not None or not op_available():
        return
    torch.library.register_fake("deepmd::dpa4c_canonical_compress")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4c_canonical_compress_backward")(
        _backward_fake
    )
    torch.library.register_fake("deepmd::dpa4c_canonical_compress_energy_force")(
        _energy_force_fake
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
        "dpa4c_canonical_compress_energy_force",
        _cpu_energy_force,
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
    atom_bias: torch.Tensor,
    spin: torch.Tensor | None = None,
    pair_table: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate compressed DPA4C from a compact canonical edge stream.

    The compact ABI carries only source indices and CSR row pointers, so the
    descriptor operator addresses edges by position and omits every mask,
    padding-type, and table-tail check that the cutoff-compacted graph already
    guarantees. The backward reuses the forward state in place and assembles
    the force and the virial itself, so the source CSR is read only for the
    per-atom virial: a graph that carries placeholders in its place yields a
    zero per-atom virial. The operator also derives the ownership mask, masks
    the node energies and forms the frame totals, so that nothing around it
    runs on the node axis.

    The forces are summed in fixed point. A frame whose energy or virial is
    not finite reports every force of its nodes, and every magnetic force, as
    not a number, and so does a frame with a force contribution of 2^30 eV/Å
    or more, which lies outside the range of the sums; its node then reports
    its energy as not a number. The check covers each contribution, so a sum
    of three or more smaller ones beyond the range of 2^31 eV/Å, which no
    configuration short of three atoms within about 0.01 Å of one atom
    reaches, is not detected.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.
    fitting
        Eligible pt_expt energy fitting network.
    graph
        Canonical graph whose ``source`` has shape ``(E,)`` in ``uint32`` and
        whose ``destination_row_ptr`` has shape ``(N + 1,)`` in ``int64``.
        Its source CSR is either complete or a two-entry placeholder.
    atype
        Flat node atom types with shape ``(N,)``; a negative type marks a
        virtual node, which contributes no energy.
    atom_bias
        Combined atomic energy bias with shape ``(ntypes,)`` in eV.
    spin
        Per-node magnetic moments with shape ``(N, 3)`` for a spin-conditioned
        descriptor, or ``None``.
    pair_table
        Series constants of an analytical pair potential with shape
        ``((T + 1) ** 2, 8)``, or ``None``. The fused operator adds its energy
        to the node energies and its slope to the edge gradient.

    Returns
    -------
    energy
        Per-frame energy with shape ``(F, 1)`` in eV, fp64.
    atom_energy
        Per-node energy with shape ``(N, 1)`` in eV, fp64, zero for every node
        outside the ownership mask whose forces lie in range.
    force
        Per-node force with shape ``(N, 3)`` in eV/Å, fp64.
    virial
        Per-frame virial with shape ``(F, 3, 3)`` in eV, fp64.
    atom_virial
        Per-node virial with shape ``(N, 3, 3)`` in eV, fp32, attributed to
        source nodes; zero when the graph carries no source CSR.
    force_mag
        Per-node magnetic force with shape ``(N, 3)`` in fp64 for a
        spin-conditioned descriptor, or an empty tensor. Both the on-site part
        and the neighbour part close inside the fused operator.
    mask
        Ownership mask with shape ``(N,)``, int32: one for a real node among
        the leading ``n_local`` nodes of its frame.

    Raises
    ------
    ValueError
        If the descriptor lies outside the configurations the compiled
        operator specializes on, or the fused fitting operator cannot
        reproduce the fitting network.
    """
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        compressed_operator_arguments,
        mega_eligible,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        ensure_registered as ensure_fitting_registered,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
        node_tile,
    )

    ensure_registered()
    ensure_fitting_registered()
    if not mega_eligible(descriptor):
        raise ValueError("model is not eligible for compact canonical DPA4C inference")

    network = fitting_operator_arguments(fitting)
    return torch.ops.deepmd.dpa4c_canonical_compress_energy_force(
        graph.edge_vec,
        graph.source,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        graph.n_node,
        graph.n_local,
        atype,
        *compressed_operator_arguments(descriptor, spin),
        int(descriptor.lmax),
        *descriptor._compression_scalars,
        network.weights,
        network.biases,
        network.residuals,
        network.head_weight,
        network.head_bias,
        atom_bias.to(torch.float64).contiguous(),
        network.activation,
        node_tile(),
        graph.edge_vec.new_empty(0) if pair_table is None else pair_table,
    )
