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
    choose_nv_nlist_method,
    is_nv_available,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)


def _grow_search_capacity(capacity: int) -> int:
    """Increase Toolkit-Ops capacity by 1.25x, rounded up (mirror nv_nlist)."""
    return (capacity * 5 + 3) // 4


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
    from nvalchemiops.torch.neighbors import (
        neighbor_list,
    )

    device = coord.device
    nf = coord.shape[0] if coord.ndim == 3 else 1
    coord = coord.reshape(nf, -1, 3)
    nloc = coord.shape[1]
    periodic = box is not None

    if nloc == 0:
        empty_i = torch.zeros((0,), dtype=torch.int64, device=device)
        empty_S = torch.zeros((0, 3), dtype=torch.int64, device=device)
        return neighbor_graph_from_ijs(
            empty_i, empty_i, empty_S, coord, box, empty_i, nloc, layout=layout
        )

    # Mirror nv_nlist.build's search setup: normalize periodic coords (a
    # differentiable lattice translation, identity gradient), flatten the batch,
    # and search all frames in ONE kernel via batch_idx/batch_ptr.
    if periodic:
        cell = box.reshape(nf, 3, 3).to(device=device, dtype=coord.dtype)
        coord = normalize_coord(coord, cell)
        pbc = torch.ones((nf, 3), dtype=torch.bool, device=device)
    else:
        cell = None
        pbc = None
    box_out = cell  # edge_vec is recomputed from these (normalized) coords

    total_atoms = nf * nloc
    positions = coord.reshape(total_atoms, 3).detach()
    batch_idx = torch.arange(nf, dtype=torch.int32, device=device).repeat_interleave(
        nloc
    )
    batch_ptr = torch.arange(nf + 1, dtype=torch.int32, device=device) * nloc
    method = choose_nv_nlist_method(nloc, periodic=periodic, device=device)
    extra_nl_kwargs: dict[str, Any] = {}
    if method == "batch_naive":
        extra_nl_kwargs["max_atoms_per_system"] = int(nloc)

    # Carry-all: grow capacity until every neighbor fits (no sel cap).
    search_capacity = max(64, nloc)
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
        max_found = int(num_neighbors.max().item()) if num_neighbors.numel() > 0 else 0
        if max_found <= search_capacity:
            break
        search_capacity = max(max_found, _grow_search_capacity(search_capacity))

    # Decode the dense matrix to a sparse (i, j, S) edge list -- Step 1 of
    # nv_nlist._matrix_to_extended_inputs. neighbor_matrix[dst, slot] = src, both
    # flattened indices in [0, total_atoms); frames are batch-isolated so a
    # neighbor shares the center's frame.
    max_neighbors = neighbor_matrix.shape[1]
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

    return neighbor_graph_from_ijs(
        center_local, src_local, shift, coord, box_out, frame_idx, nloc, layout=layout
    )
