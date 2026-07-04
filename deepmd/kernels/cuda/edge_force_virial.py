# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused force / virial assembly of the graph lower.

The CUDA operator ``deepmd::edge_force_virial`` (see
``source/op/pt/edge_force_virial.cu``) scatters the per-edge energy gradient
``g_e = dE/d(edge_vec)`` into per-atom force, per-atom virial (optional) and
per-frame virial through destination/source CSR reductions, replacing the array-API
:func:`~deepmd.dpmodel.utils.neighbor_graph.edge_force_virial` chain of
``index_add`` / outer-product / ``segment_sum`` kernels. It is
descriptor-agnostic: any graph-lowered model whose force path differentiates
the energy w.r.t. ``edge_vec`` can dispatch here.

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

__all__ = [
    "edge_force_virial",
    "ensure_registered",
    "op_available",
]

_registered = False


def op_available() -> bool:
    """Whether the C++ ``deepmd::edge_force_virial`` op is loaded."""
    op = getattr(torch.ops.deepmd, "edge_force_virial", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def _fake(
    g_e: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    node_capacity: int,
    want_atom_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_frame = n_node_per_frame.shape[0]
    return (
        g_e.new_empty(node_capacity, 3),
        g_e.new_empty(node_capacity if want_atom_virial else 0, 3, 3),
        g_e.new_empty(n_frame, 3, 3),
    )


def _cpu(
    g_e: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    node_capacity: int,
    want_atom_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from deepmd.dpmodel.utils.neighbor_graph import edge_force_virial as reference

    force, atom_virial, virial = reference(
        g_e,
        edge_vec,
        edge_index,
        edge_mask,
        n_node_per_frame,
        node_capacity=node_capacity,
    )
    if not want_atom_virial:
        atom_virial = atom_virial.new_zeros(0, 3, 3)
    return force, atom_virial, virial


_cpu_library: torch.library.Library | None = None


def ensure_registered() -> None:
    """Register the fake and CPU implementations for the op.

    Idempotent; a no-op when the C++ operator library is not loaded.
    """
    global _registered, _cpu_library
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::edge_force_virial")(_fake)
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("edge_force_virial", _cpu, "CPU")
    _registered = True


def edge_force_virial(
    g_e: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    n_node_per_frame: torch.Tensor,
    node_capacity: int,
    want_atom_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """
    ensure_registered()
    return torch.ops.deepmd.edge_force_virial(
        g_e.contiguous(),
        edge_vec.contiguous(),
        edge_index.contiguous(),
        edge_mask.contiguous(),
        destination_order.contiguous(),
        destination_row_ptr.contiguous(),
        source_row_ptr.contiguous(),
        source_order.contiguous(),
        n_node_per_frame,
        node_capacity,
        want_atom_virial,
    )
