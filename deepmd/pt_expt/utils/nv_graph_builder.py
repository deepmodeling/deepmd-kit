# SPDX-License-Identifier: LGPL-3.0-or-later
"""Carry-all NeighborGraph builder backed by nvalchemiops (GPU cell list).

World-2 counterpart of :mod:`deepmd.pt.utils.nv_nlist`: instead of building the
dense extended quartet, it decodes nvalchemiops' dense
``(total_atoms, max_neighbors)`` neighbor matrix into flat per-frame local
``(i, j, S, nframe_id)`` and delegates to the array-API
:func:`~deepmd.dpmodel.utils.neighbor_graph.neighbor_graph_from_ijs`, which
recomputes ``edge_vec`` differentiably from the (normalized) coordinates.

Unlike the vesin builder, nvalchemiops batches natively over frames via
``batch_idx``/``batch_ptr`` -- a single GPU kernel handles all ``nf`` frames,
so there is NO per-frame Python loop. CUDA-only ⇒ this module lives in pt_expt.

The matrix decode mirrors :func:`deepmd.pt.utils.nv_nlist._matrix_to_extended_inputs`
(the authoritative, tested extraction) but stops at the sparse ``(i, j, S)``
edge list rather than materializing the extended-atom contract.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
    neighbor_graph_from_ijs,
)
from deepmd.pt.utils.nv_nlist import (
    _input_device_context,
    choose_nv_nlist_method,
    is_nv_available,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)


def _grow_search_capacity(capacity: int) -> int:
    """Increase Toolkit-Ops capacity by 1.25x, rounded up (mirror nv_nlist)."""
    return (capacity * 5 + 3) // 4


def nv_matrix_to_ijs(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
    nloc: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode nvalchemiops' dense neighbor matrix to a sparse edge list.

    Pure torch and device-agnostic (CPU-runnable), so the regression-prone
    index arithmetic is unit-testable on the default CI without CUDA — the
    GPU ``neighbor_list`` search itself stays behind the opt-in CUDA suite.
    Step 1 of :func:`deepmd.pt.utils.nv_nlist._matrix_to_extended_inputs`.

    Parameters
    ----------
    neighbor_matrix
        (total_atoms, max_neighbors) int; ``neighbor_matrix[dst, slot] = src``,
        both flattened batch indices in ``[0, total_atoms)``. Frames are
        batch-isolated: a neighbor always shares its center's frame.
    num_neighbors
        (total_atoms,) int, valid slot count per center.
    shifts
        (total_atoms, max_neighbors, 3) int periodic image shifts per slot.
    nloc
        Atoms per frame (``total_atoms = nf * nloc``).

    Returns
    -------
    center_local
        (E,) int64 per-frame local center index ``i`` (``dst % nloc``).
    src_local
        (E,) int64 per-frame local neighbor index ``j`` (``src % nloc``).
    shift
        (E, 3) int64 periodic image shift ``S``.
    frame_idx
        (E,) int64 frame of each edge (``dst // nloc``).
    """
    device = neighbor_matrix.device
    total_atoms, max_neighbors = neighbor_matrix.shape
    slot = torch.arange(max_neighbors, dtype=torch.long, device=device).expand(
        total_atoms, max_neighbors
    )
    valid = (slot < num_neighbors.unsqueeze(1)).reshape(-1)
    edge_idx = torch.nonzero(valid, as_tuple=False).flatten()

    dst = edge_idx // max_neighbors  # flattened center
    src = neighbor_matrix.reshape(-1).index_select(0, edge_idx).to(torch.int64)
    shift = shifts.reshape(-1, 3).index_select(0, edge_idx).to(torch.int64)
    frame_idx = (dst // nloc).to(torch.int64)  # frame of the edge
    center_local = (dst % nloc).to(torch.int64)  # i = center
    src_local = (src % nloc).to(torch.int64)  # j = neighbor
    return center_local, src_local, shift, frame_idx


def nv_search_matrix(
    coord: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    start_capacity: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the nvalchemiops neighbor search and return the raw matrix output.

    Encapsulates the full search pipeline: ``_input_device_context`` pinning,
    periodic coordinate normalization, batch tensor construction, and the
    grow-until-fit capacity loop.  This is the single authoritative nv search;
    :class:`~deepmd.pt.utils.nv_nlist.NvNeighborList` delegates here so the
    search logic is maintained in exactly one place.

    Parameters
    ----------
    coord : (nf, nloc, 3) local coordinates (already reshaped).
    box : (nf, 3, 3) simulation cell, or ``None`` for non-periodic.
    rcut : cutoff radius.
    start_capacity : initial max-neighbor capacity; grown automatically when
        any atom has more neighbors than the current capacity.

    Returns
    -------
    coord : (nf, nloc, 3) coordinates, normalized in-cell if periodic.
    cell : (nf, 3, 3) float box, or ``None`` for non-periodic.
    neighbor_matrix : (total_atoms, capacity) int neighbor matrix.
    num_neighbors : (total_atoms,) valid neighbor count per center.
    shifts : (total_atoms, capacity, 3) int periodic image shifts.
    """
    from nvalchemiops.torch.neighbors import (
        neighbor_list,
    )

    device = coord.device
    nf = coord.shape[0]
    nloc = coord.shape[1]
    periodic = box is not None

    with _input_device_context(device):
        if periodic:
            cell = box.reshape(nf, 3, 3).to(device=device, dtype=coord.dtype)
            coord = normalize_coord(coord, cell)
            pbc = torch.ones((nf, 3), dtype=torch.bool, device=device)
        else:
            cell = None
            pbc = None

        total_atoms = nf * nloc
        positions = coord.reshape(total_atoms, 3).detach()
        batch_idx = torch.arange(
            nf, dtype=torch.int32, device=device
        ).repeat_interleave(nloc)
        batch_ptr = torch.arange(nf + 1, dtype=torch.int32, device=device) * nloc
        method = choose_nv_nlist_method(nloc, periodic=periodic, device=device)
        extra_nl_kwargs: dict[str, Any] = {}
        if method == "batch_naive":
            extra_nl_kwargs["max_atoms_per_system"] = int(nloc)

        search_capacity = start_capacity
        while True:
            nlist_result = neighbor_list(
                positions,
                float(rcut),
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                method=method,
                max_neighbors=int(search_capacity),
                return_neighbor_list=False,
                wrap_positions=False,
                **extra_nl_kwargs,
            )
            if len(nlist_result) == 2:
                neighbor_matrix, num_neighbors = nlist_result
                shifts = torch.zeros(
                    (*neighbor_matrix.shape, 3), dtype=torch.int32, device=device
                )
            else:
                neighbor_matrix, num_neighbors, shifts = nlist_result
            max_found = (
                int(num_neighbors.max().item()) if num_neighbors.numel() > 0 else 0
            )
            if max_found <= search_capacity:
                break
            search_capacity = max(max_found, _grow_search_capacity(search_capacity))

    return coord, cell, neighbor_matrix, num_neighbors, shifts


def build_neighbor_graph_nv(
    coord: Any,
    atype: Any,
    box: Any | None,
    rcut: float,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Build a CARRY-ALL NeighborGraph using nvalchemiops' GPU cell list.

    Parameters
    ----------
    coord
        (nf, nloc, 3) or (nf, nloc*3) local coordinates (CUDA tensor).
    atype
        (nf, nloc) local atom types (carried for API parity).
    box
        (nf, 3, 3) simulation cell, or ``None`` for non-periodic.
    rcut
        cutoff radius.
    layout
        edge-axis length policy; ``None`` => dynamic with ``min_edges`` guards.

    Returns
    -------
    graph
        The carry-all :class:`NeighborGraph` over the LOCAL atoms, ``edge_vec``
        recomputed differentiably from the (normalized) ``coord``/``box``.

    Raises
    ------
    ImportError
        if ``nvalchemi-toolkit-ops`` (CUDA) is not installed.
    """
    if not is_nv_available():
        raise ImportError(
            "build_neighbor_graph_nv requires nvalchemi-toolkit-ops (CUDA); "
            "install with `pip install nvalchemi-toolkit-ops` or use "
            "neighbor_graph_method='dense'."
        )

    device = coord.device
    nf = coord.shape[0] if coord.ndim == 3 else 1
    coord = coord.reshape(nf, -1, 3)
    nloc = coord.shape[1]

    if nloc == 0:
        empty_i = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_S = torch.zeros((0, 3), dtype=torch.int64, device=device)
        return neighbor_graph_from_ijs(
            empty_i, empty_i, empty_S, coord, box, empty_i, nloc, layout=layout
        )

    # Carry-all: grow capacity until every neighbor fits (no sel cap).
    # NOTE: unlike the vesin builder (which searches the ORIGINAL coords --
    # vesin handles unwrapped positions natively), nvalchemiops requires
    # in-cell positions, so BOTH the search and the edge_vec recomputation use
    # the normalized coords; S then matches the coords the search actually saw.
    coord, cell, neighbor_matrix, num_neighbors, shifts = nv_search_matrix(
        coord, box, rcut, start_capacity=max(64, nloc)
    )
    box_out = cell  # edge_vec is recomputed from these (normalized) coords

    # Decode the dense matrix to a sparse (i, j, S) edge list (CPU-testable
    # helper; see nv_matrix_to_ijs).
    center_local, src_local, shift, frame_idx = nv_matrix_to_ijs(
        neighbor_matrix, num_neighbors, shifts, nloc
    )

    # virtual atoms (atype < 0) are excluded as centers AND neighbors — the
    # World-2 builder contract shared with the dense reference builder; the
    # geometric search above cannot know about them.
    at = torch.as_tensor(atype, device=device).reshape(nf, nloc)
    keep = (at[frame_idx, center_local] >= 0) & (at[frame_idx, src_local] >= 0)
    center_local, src_local = center_local[keep], src_local[keep]
    shift, frame_idx = shift[keep], frame_idx[keep]

    return neighbor_graph_from_ijs(
        center_local, src_local, shift, coord, box_out, frame_idx, nloc, layout=layout
    )
