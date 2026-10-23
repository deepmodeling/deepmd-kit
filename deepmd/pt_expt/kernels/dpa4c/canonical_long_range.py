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
