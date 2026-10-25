# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact canonical deployment of compressed DPA4C with a long-range module."""

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


def fused_long_range_eligible(long_range: Any) -> bool:
    """Whether the fused long-range operator evaluates a long-range module.

    The operator evaluates a charge network of one channel that the fused
    fitting kernels reproduce, beside the compressed descriptor and the energy
    fitting that the compact canonical lower already requires; a module whose
    charge network they do not reproduce, or a library without the operator,
    takes :func:`dpa4c_canonical_long_range_energy_force`.
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        long_range_op_available,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_eligible,
    )

    return long_range_op_available() and fitting_eligible(long_range.charge_net)


def dpa4c_canonical_compress_long_range_energy_force(
    descriptor: Any,
    fitting: Any,
    long_range: Any,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    coord: torch.Tensor,
    communicator: torch.Tensor | None,
    box: torch.Tensor | None,
    atom_bias: torch.Tensor,
    fitting_routes: list[int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate compressed DPA4C and its long-range module in one operator.

    The operator evaluates the descriptor of every node, the charges of the
    module from it, the energy fitting, the long range of the charges and one
    descriptor backward from the cotangents of both networks, and assembles
    the forces in fixed point like
    :func:`~deepmd.pt_expt.kernels.dpa4c.canonical.dpa4c_canonical_compress_energy_force`,
    so the force assembly reads the source CSR only for the per-atom virial.
    It derives the ownership mask from the frame counts, and the charged
    nodes of the module are the owned real nodes.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor without a spin condition.
    fitting
        Eligible pt_expt energy fitting network.
    long_range
        Long-range module for which :func:`fused_long_range_eligible` holds.
    graph
        Canonical graph as for
        :func:`~deepmd.pt_expt.kernels.dpa4c.canonical.dpa4c_canonical_compress_energy_force`;
        its source CSR is either complete or a two-entry placeholder, which
        declines the per-atom virial.
    atype
        Flat node atom types with shape ``(N,)``; a negative type marks a
        virtual node, which contributes no energy and carries no charge.
    coord
        Node positions with shape ``(N, 3)`` in Å.
    communicator
        The address of the MPI_Comm handle of a frame decomposed into
        domains, as an int64 CPU tensor of one element, or zero or ``None``
        for a single domain.
    box
        The cells of the frames with shape ``(nf, 9)``, float64 on the CPU, a
        zero row marking a complete non-periodic frame, or ``None`` when no
        frame is periodic.
    atom_bias
        Combined atomic energy bias with shape ``(ntypes,)`` in eV.
    fitting_routes
        Fitting-route table of the energy fitting, as for the fused operator
        without a long-range module, or ``None`` for the measured table.

    Returns
    -------
    energy
        Per-frame energy with shape ``(F, 1)`` in eV, fp64.
    atom_energy
        Per-node energy with shape ``(N, 1)`` in eV, fp64.
    force
        Per-node force with shape ``(N, 3)`` in eV/Å, fp64.
    virial
        Per-frame virial with shape ``(F, 3, 3)`` in eV, fp64.
    atom_virial
        Per-node virial with shape ``(N, 3, 3)`` in eV, fp32; zero when the
        graph carries no source CSR.
    mask
        Ownership mask with shape ``(N,)``, int32.
    """
    from deepmd.dpmodel.long_range.kernel import (
        COULOMB_CONSTANT,
    )
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        ensure_registered,
    )
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        compressed_operator_arguments,
    )
    from deepmd.pt_expt.kernels.fitting_routes import fitting_routes as measured_routes
    from deepmd.pt_expt.kernels.graph_fitting import (
        ensure_registered as ensure_fitting_registered,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
        node_tile,
    )
    from deepmd.pt_expt.kernels.long_range import (
        kernel_operator_arguments,
    )

    ensure_registered()
    ensure_fitting_registered()
    network = fitting_operator_arguments(fitting)
    charge_net = long_range.charge_net
    charges = fitting_operator_arguments(charge_net)
    total_charge, channel = long_range.operator_constraint(graph.n_node.shape[0], coord)
    kernel, coefficient = kernel_operator_arguments(long_range.kernel)
    if communicator is None:
        communicator = torch.zeros(1, dtype=torch.int64, device="cpu")
    routes = measured_routes(fitting) if fitting_routes is None else fitting_routes
    return torch.ops.deepmd.dpa4c_canonical_compress_long_range_energy_force(
        graph.edge_vec,
        graph.source,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        graph.n_node,
        graph.n_local,
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
        node_tile(),
        charges.weights,
        charges.biases,
        charges.residuals,
        charges.head_weight,
        charges.head_bias,
        charge_net.bias_atom_e[:, 0].to(torch.float64).contiguous(),
        charges.activation,
        coord.to(torch.float64).contiguous(),
        total_charge,
        # A module that does not serve periodic systems sums every frame
        # directly.
        box if long_range.periodic else None,
        coefficient.to(coord.device),
        communicator,
        kernel,
        channel,
        COULOMB_CONSTANT,
        long_range.pme_order,
        long_range.mesh_spacing() if long_range.periodic else 1.0,
        # No table is passed as None rather than as an empty list: the
        # AOTInductor proxy executor of torch 2.13 and earlier hands a
        # serialized empty int list to the operator as None.
        list(routes) if routes else None,
    )


def dpa4c_canonical_long_range_energy_force(
    descriptor: Any,
    fitting: Any,
    long_range: Any,
    graph: CanonicalGraph,
    atype: torch.Tensor,
    coord: torch.Tensor,
    communicator: torch.Tensor | None,
    box: torch.Tensor | None,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate compressed DPA4C and its long-range module from a compact graph.

    The descriptor operator runs once, and its output feeds both the fused
    energy fitting and the charge network of the long-range module. The
    derivative of the energy with respect to the descriptor collects the
    fitting term and the response of the charges, so a single descriptor
    backward yields the edge gradient of both; the long-range force at fixed
    charges is added to the assembled force.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor without a spin condition.
    fitting
        Eligible pt_expt energy fitting network.
    long_range
        Long-range module whose ``operator_eligible()`` holds.
    graph
        Canonical graph whose ``source`` has shape ``(E,)`` in ``uint32`` and
        whose ``destination_row_ptr`` has shape ``(N + 1,)`` in ``int64``.
    atype
        Flat node atom types with shape ``(N,)``.
    coord
        Node positions with shape ``(N, 3)`` in Å.
    communicator
        The address of the MPI_Comm handle of a frame decomposed into
        domains, one per rank, as an int64 CPU tensor of one element, or zero
        or ``None`` for a single domain. Across domains the owned nodes of
        every rank make up the frame.
    box
        The cells of the frames with shape ``(nf, 9)``, float64 on the CPU,
        a zero row marking a complete non-periodic frame, or ``None`` when no
        frame is periodic.
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
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        ensure_registered,
    )
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        compressed_operator_arguments,
        fitting_energy_and_gradient,
    )
    from deepmd.pt_expt.kernels.edge_force_virial import (
        canonical_edge_force_virial,
    )
    from deepmd.pt_expt.kernels.edge_force_virial import (
        ensure_registered as ensure_force_registered,
    )
    from deepmd.pt_expt.kernels.edge_force_virial import (
        frame_scalar_sum,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        ensure_registered as ensure_fitting_registered,
    )

    ensure_registered()
    ensure_fitting_registered()
    ensure_force_registered()
    operator_args = (
        graph.source,
        graph.destination_row_ptr,
        atype,
        *compressed_operator_arguments(descriptor),
        int(descriptor.lmax),
        *descriptor._compression_scalars,
    )
    node_descriptor, state = torch.ops.deepmd.dpa4c_canonical_compress(
        graph.edge_vec, *operator_args
    )
    # The charge network reads the descriptor before the fused fitting
    # overwrites it with its own cotangent.
    (
        long_range_energy,
        long_range_gradient,
        long_range_force,
        long_range_atom_virial,
        long_range_virial,
    ) = long_range.energy_force_from_coord(
        node_descriptor, atype, ownership, graph.n_node, coord, communicator, box
    )
    atom_energy, descriptor_gradient = fitting_energy_and_gradient(
        fitting, node_descriptor, atype, ownership, atom_bias
    )
    edge_gradient, _, _ = torch.ops.deepmd.dpa4c_canonical_compress_backward_inplace(
        descriptor_gradient + long_range_gradient,
        state,
        graph.edge_vec,
        *operator_args,
    )
    force, atom_virial, virial, _ = canonical_edge_force_virial(
        edge_gradient,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        graph.n_node,
        graph.edge_vec.new_empty(0),
        atype.shape[0],
        do_atomic_virial,
    )
    # Nodes outside the ownership carry no charge and hence no long-range
    # energy; the halo nodes of a periodic frame are images, which the
    # reciprocal-space sum holds itself.
    atom_energy = atom_energy + long_range_energy
    energy = frame_scalar_sum(atom_energy, graph.n_node)
    virial = virial + long_range_virial.reshape(-1, 3, 3)
    if do_atomic_virial:
        atom_virial = atom_virial + long_range_atom_virial.reshape(-1, 3, 3)
    return energy, atom_energy, force + long_range_force, virial, atom_virial
