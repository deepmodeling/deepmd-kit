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
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
    apply_pair_exclusion,
    attach_edge_csr,
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
    node_index: torch.Tensor | None = None,
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
        both flattened search indices in ``[0, total_atoms)``. Frames are
        batch-isolated: a neighbor always shares its center's frame.
    num_neighbors
        (total_atoms,) int, valid slot count per center.
    shifts
        (total_atoms, max_neighbors, 3) int periodic image shifts per slot.
    nloc
        Atoms per frame of the rectangular batch the indices are reported on.
    node_index
        (total_atoms,) int, position of each searched atom in that rectangular
        batch. Given when the search ran over a subset of the batch, such as
        its real atoms alone; ``None`` when it ran over every slot, in which
        case the search index and the batch position coincide.

    Returns
    -------
    center_local
        (E,) int64 per-frame local center index ``i``.
    src_local
        (E,) int64 per-frame local neighbor index ``j``.
    shift
        (E, 3) int64 periodic image shift ``S``.
    frame_idx
        (E,) int64 frame of each edge.
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
    if node_index is not None:
        # Lift both endpoints from the search axis back onto the rectangular
        # batch, so that the frame and local indices below are the ones every
        # other builder reports and the caller needs no special case.
        dst = node_index.index_select(0, dst)
        src = node_index.index_select(0, src)
    frame_idx = (dst // nloc).to(torch.int64)  # frame of the edge
    center_local = (dst % nloc).to(torch.int64)  # i = center
    src_local = (src % nloc).to(torch.int64)  # j = neighbor
    return center_local, src_local, shift, frame_idx


def nv_search_matrix(
    coord: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    start_capacity: int,
    node_index: torch.Tensor | None = None,
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
    node_index : (total_atoms,) positions, in the flattened batch, of the atoms
        to search over, frame-major and ascending. ``None`` searches every slot.
        The search itself is ragged -- nvalchemiops takes a flat position array
        with explicit per-frame bounds -- so restricting it to the real atoms of
        a padded batch keeps the phantom slots out of a cost that grows with the
        square of the frame width.

    Returns
    -------
    coord : (nf, nloc, 3) coordinates, normalized in-cell if periodic.
    cell : (nf, 3, 3) float box, or ``None`` for non-periodic.
    neighbor_matrix : (total_atoms, capacity) int neighbor matrix, indexed on
        the searched atoms rather than on the batch slots.
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

        positions = coord.reshape(nf * nloc, 3).detach()
        if node_index is None:
            batch_idx = torch.arange(
                nf, dtype=torch.int32, device=device
            ).repeat_interleave(nloc)
            batch_ptr = torch.arange(nf + 1, dtype=torch.int32, device=device) * nloc
            widest_frame = nloc
        else:
            positions = positions.index_select(0, node_index)
            batch_idx = (node_index // nloc).to(torch.int32)
            counts = torch.bincount(batch_idx.to(torch.int64), minlength=nf)
            batch_ptr = torch.zeros(nf + 1, dtype=torch.int32, device=device)
            batch_ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
            widest_frame = int(counts.max()) if nf > 0 else 0
        method = choose_nv_nlist_method(widest_frame, periodic=periodic, device=device)
        extra_nl_kwargs: dict[str, Any] = {}
        if method == "batch_naive":
            extra_nl_kwargs["max_atoms_per_system"] = int(widest_frame)

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
    *,
    with_csr: bool = False,
    canonicalize: bool = False,
    pair_excl: PairExcludeMask | None = None,
    compact: bool = False,
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
    with_csr
        Whether to construct destination/source CSR views for a consumer that
        requires edge-grouped reductions.
    canonicalize
        Whether to reorder every edge field into destination-major form. Implies
        ``with_csr=True``.
    pair_excl
        Optional :class:`~deepmd.dpmodel.utils.neighbor_graph.graph.PairExcludeMask`
        for model-level ``pair_exclude_types``. When given,
        :func:`apply_pair_exclusion` is applied after the geometric search. ``None``
        (default) leaves all geometrically valid edges present.
    compact
        Passed to :func:`apply_pair_exclusion`; see that function for details.
        Ignored when ``pair_excl`` is ``None``.

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
    from nvalchemiops.neighbors.neighbor_utils import (
        estimate_max_neighbors,
    )

    device = coord.device
    nf, nloc = atype.shape[:2]
    coord = coord.reshape(nf, nloc, 3)
    periodic = box is not None

    if nloc == 0:
        empty_i = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_S = torch.zeros((0, 3), dtype=torch.int64, device=device)
        return neighbor_graph_from_ijs(
            empty_i,
            empty_i,
            empty_S,
            coord,
            box,
            empty_i,
            torch.full((nf,), nloc, dtype=torch.int64, device=device),
            layout=layout,
            with_csr=with_csr,
            canonicalize=canonicalize,
        )

    # Carry-all: grow capacity until every neighbor fits (no sel cap).
    # NOTE: unlike the vesin builder (which searches the ORIGINAL coords --
    # vesin handles unwrapped positions natively), nvalchemiops requires
    # in-cell positions, so BOTH the search and the edge_vec recomputation use
    # the normalized coords; S then matches the coords the search actually saw.
    # The 0.25 density preserves a 25% margin over the estimator's 0.2
    # baseline without using the safety_factor argument deprecated in Ops 0.4.
    initial_capacity = max(
        64,
        estimate_max_neighbors(float(rcut), atomic_density=0.25),
    )
    # Virtual atoms (atype < 0) are excluded as centers AND neighbors -- the
    # World-2 builder contract shared with the dense reference builder. Here
    # they are withheld from the search rather than filtered out of its result:
    # the phantom slots of a mixed-nloc batch would otherwise widen every frame
    # the search sees, and its cost grows with the square of that width.
    atype_flat = torch.as_tensor(atype, device=device).reshape(nf * nloc)
    node_index = torch.nonzero(atype_flat >= 0, as_tuple=False).flatten()
    coord, cell, neighbor_matrix, num_neighbors, shifts = nv_search_matrix(
        coord, box, rcut, start_capacity=initial_capacity, node_index=node_index
    )
    box_out = cell  # edge_vec is recomputed from these (normalized) coords

    # Decode the dense matrix to a sparse (i, j, S) edge list, lifted back onto
    # the rectangular batch (CPU-testable helper; see nv_matrix_to_ijs).
    center_local, src_local, shift, frame_idx = nv_matrix_to_ijs(
        neighbor_matrix, num_neighbors, shifts, nloc, node_index=node_index
    )

    graph = neighbor_graph_from_ijs(
        center_local,
        src_local,
        shift,
        coord,
        box_out,
        frame_idx,
        torch.full((nf,), nloc, dtype=torch.int64, device=device),
        layout=layout,
    )
    if pair_excl is not None:
        graph = apply_pair_exclusion(graph, atype_flat, pair_excl, compact=compact)
    if with_csr or canonicalize:
        graph = attach_edge_csr(graph, nf * nloc, canonicalize=canonicalize)
    return graph
