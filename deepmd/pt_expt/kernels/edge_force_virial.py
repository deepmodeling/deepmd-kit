# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused force / virial assembly of the graph lower.

The operator ``deepmd::edge_force_virial`` scatters the per-edge energy
gradient ``g_e = dE/d(edge_vec)`` into per-atom force, per-atom virial
(optional) and per-frame virial through destination/source CSR reductions,
replacing the array-API
:func:`~deepmd.dpmodel.utils.neighbor_graph.edge_force_virial` chain of
``index_add`` / outer-product / ``segment_sum`` kernels. It is
descriptor-agnostic: any graph-lowered model whose force path differentiates
the energy w.r.t. ``edge_vec`` can dispatch here. The CUDA kernel is
``source/op/pt/edge_force_virial.cu`` and the CPU kernel
``source/op/pt/edge_force_virial_cpu.cc``.

Usage and pitfalls
------------------
* ``node_capacity`` is declared ``SymInt`` in the op schema. With a plain
  ``int`` the traced graph would specialize the padded node count to the
  trace-time value and a resized deployment would read out of bounds; the
  ``SymInt`` keeps it symbolic through ``make_fx`` / ``torch.export``.
* The op is dispatched downstream of a ``torch.autograd.grad`` call, so it
  needs no registered backward of its own; only the fake and CPU
  implementations are required for tracing.
* When the atomic virial is not requested the op returns an empty
  ``(0, 3, 3)`` tensor instead of skipping the output (the schema is static);
  the caller maps it to ``None``.
* CSR rows describe topology rather than validity. Masked entries may remain
  inside a row, so the kernel applies ``edge_mask`` to both incidence
  reductions.
"""

import torch

from deepmd.pt_expt.kernels.utils import (
    operator_available,
)

__all__ = [
    "canonical_edge_force_virial",
    "canonical_op_available",
    "edge_force_virial",
    "ensure_registered",
    "frame_scalar_sum",
    "frame_scalar_sum_available",
    "op_available",
]


def op_available() -> bool:
    """Whether the backend device carries ``deepmd::edge_force_virial``."""
    return operator_available("edge_force_virial")


def canonical_op_available() -> bool:
    """Whether the backend device carries the compact canonical force operator."""
    return operator_available("canonical_edge_force_virial")


def frame_scalar_sum_available() -> bool:
    """Whether the backend device carries ``deepmd::frame_scalar_sum``."""
    return operator_available("frame_scalar_sum")


def _frame_scalar_sum_fake(
    node_scalar: torch.Tensor,
    n_node_per_frame: torch.Tensor,
) -> torch.Tensor:
    return node_scalar.new_empty(n_node_per_frame.shape[0], 1)


def frame_scalar_sum(
    node_scalar: torch.Tensor,
    n_node_per_frame: torch.Tensor,
) -> torch.Tensor:
    """Sum a node-major scalar over the node segment of each frame.

    Parameters
    ----------
    node_scalar : torch.Tensor
        Per-node scalar with shape ``(N, 1)``.
    n_node_per_frame : torch.Tensor
        Node count of each frame with shape ``(F,)``. The frames occupy
        contiguous spans of the node axis in this order.

    Returns
    -------
    torch.Tensor
        Per-frame total with shape ``(F, 1)`` and the input dtype.

    Notes
    -----
    Nodes past ``sum(n_node_per_frame)`` are padding of the flat node axis and
    contribute to no frame.
    """
    ensure_registered()
    return torch.ops.deepmd.frame_scalar_sum(node_scalar, n_node_per_frame)


def _has_spin_cotangent(
    edge_spin_gradient: torch.Tensor,
) -> bool:
    """Identify a spin cotangent by its tensor rank."""
    return edge_spin_gradient.ndim == 2


def _fake(
    g_e: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    source_row_ptr: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    edge_spin_gradient: torch.Tensor,
    node_capacity: int,
    want_atom_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_frame = n_node_per_frame.shape[0]
    return (
        g_e.new_empty(node_capacity, 3),
        g_e.new_empty(node_capacity if want_atom_virial else 0, 3, 3),
        g_e.new_empty(n_frame, 3, 3),
        g_e.new_empty(node_capacity, 3)
        if _has_spin_cotangent(edge_spin_gradient)
        else g_e.new_empty(0),
    )


def _canonical_fake(
    g_e: torch.Tensor,
    edge_vec: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    edge_spin_gradient: torch.Tensor,
    node_capacity: int,
    want_atom_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del edge_vec, destination_row_ptr, source_row_ptr, source_order
    n_frame = n_node_per_frame.shape[0]
    return (
        g_e.new_empty(node_capacity, 3),
        g_e.new_empty(node_capacity if want_atom_virial else 0, 3, 3),
        g_e.new_empty(n_frame, 3, 3),
        g_e.new_empty(node_capacity, 3)
        if _has_spin_cotangent(edge_spin_gradient)
        else g_e.new_empty(0),
    )


_registered = False


def ensure_registered() -> None:
    """Register the meta implementations the export tracer needs.

    Both devices implement the assembly in C++, so only the shapes are
    described here. Idempotent; a no-op when the operator library is not
    loaded.
    """
    global _registered
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::edge_force_virial")(_fake)
    if canonical_op_available():
        torch.library.register_fake("deepmd::canonical_edge_force_virial")(
            _canonical_fake
        )
    if frame_scalar_sum_available():
        torch.library.register_fake("deepmd::frame_scalar_sum")(_frame_scalar_sum_fake)
    _registered = True


def edge_force_virial(
    g_e: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    source_row_ptr: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    edge_spin_gradient: torch.Tensor,
    node_capacity: int,
    want_atom_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble force and virial from the per-edge energy gradient.

    Matches the array-API reference up to floating summation order:
    ``force[k] = sum_{dst=k} g_e - sum_{src=k} g_e``, the atom virial is
    attributed full-to-src as ``-g_e (x) edge_vec``, and the per-frame virial
    reduces the (implicit) atom virial over each frame's atoms.

    Parameters
    ----------
    g_e : torch.Tensor
        Per-edge energy gradient with shape (E, 3).
    edge_vec : torch.Tensor
        Per-edge displacement with shape (E, 3), same dtype as ``g_e``.
    edge_index : torch.Tensor
        ``[src, dst]`` node endpoints with shape (2, E), int64.
    edge_mask : torch.Tensor
        Valid-edge mask with shape (E,), bool.
    destination_order, source_order : torch.Tensor
        Edge permutations grouped by destination/source with shape (E,), int32
        or int64.
    destination_row_ptr, source_row_ptr : torch.Tensor
        Destination/source CSR offsets with shape (N + 1,), int64.
    n_node_per_frame : torch.Tensor
        Per-frame node counts with shape (nf,), int64.
    edge_spin_gradient : torch.Tensor
        Per-edge magnetic cotangent with shape (E, 3), or a rank-one empty
        sentinel when the model carries no magnetic degree of freedom.
    node_capacity : int
        Padded node-axis size ``N`` (may be a ``SymInt`` under tracing).
    want_atom_virial : bool
        Whether to materialize the per-atom virial.

    Returns
    -------
    force : torch.Tensor
        Per-atom force with shape (N, 3).
    atom_virial : torch.Tensor
        Per-atom virial with shape (N, 3, 3), or an empty (0, 3, 3) tensor
        when not requested.
    virial : torch.Tensor
        Per-frame virial with shape (nf, 3, 3).
    magnetic_force : torch.Tensor
        Per-source total of the magnetic cotangent with shape (N, 3), or a
        rank-one empty sentinel when no spin cotangent was supplied.
    """
    ensure_registered()
    return torch.ops.deepmd.edge_force_virial(
        g_e.contiguous(),
        edge_vec.contiguous(),
        edge_index.contiguous(),
        edge_mask.contiguous(),
        destination_order.contiguous(),
        destination_row_ptr.contiguous(),
        source_order.contiguous(),
        source_row_ptr.contiguous(),
        n_node_per_frame,
        edge_spin_gradient.contiguous(),
        node_capacity,
        want_atom_virial,
    )


def canonical_edge_force_virial(
    g_e: torch.Tensor,
    edge_vec: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    edge_spin_gradient: torch.Tensor,
    node_capacity: int,
    want_atom_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble force and virial from a compact canonical edge stream.

    Parameters
    ----------
    g_e
        Per-edge energy gradient with shape ``(S, 3)``.
    edge_vec
        Per-edge displacement with shape ``(S, 3)``.
    destination_row_ptr, source_row_ptr
        Destination and source CSR offsets with shape ``(N + 1,)``.
    source_order
        Edge storage positions grouped by source with shape ``(S,)``.
    n_node_per_frame
        Per-frame node counts with shape ``(nf,)``.
    edge_spin_gradient
        Per-edge magnetic cotangent with shape ``(E, 3)``, or a rank-one
        empty sentinel when the model carries no magnetic degree of freedom.
    node_capacity
        Flat node count ``N``.
    want_atom_virial
        Whether to materialize the per-node virial.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Force, optional atom virial, frame virial, and the per-source total of
        the magnetic cotangent, the last a rank-one empty sentinel when no
        spin cotangent was supplied.
    """
    ensure_registered()
    return torch.ops.deepmd.canonical_edge_force_virial(
        g_e.contiguous(),
        edge_vec.contiguous(),
        destination_row_ptr.contiguous(),
        source_row_ptr.contiguous(),
        source_order.contiguous(),
        n_node_per_frame,
        edge_spin_gradient.contiguous(),
        node_capacity,
        want_atom_virial,
    )
