# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sparse ``(i, j, S)`` edge-list converter to :class:`NeighborGraph`.

``neighbor_graph_from_ijs`` is the canonical sparse converter: it takes an
already-built sparse edge list -- per-edge center ``i``, neighbor ``j`` (both
per-frame LOCAL indices in ``[0, nloc)``) and integer periodic-image shift ``S``
-- and emits a :class:`NeighborGraph` whose ``edge_vec`` is recomputed
DIFFERENTIABLY from ``coord``/``box`` (it never trusts the builder's distance
vectors). It is the format-conversion step shared by every O(N) search backend
(ASE/vesin/LAMMPS): a backend searches, then hands its ``(i, j, S)`` here.

Convention (matching :mod:`...graph`): ``edge_index = [src, dst]`` with
``src = j`` (neighbor's local owner), ``dst = i`` (center), and
``edge_vec = r_j + S @ box - r_i`` (neighbor image minus center).
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import array_api_compat

from .csr import (
    build_edge_csr,
)
from .graph import (
    GraphLayout,
    NeighborGraph,
    pad_and_guard_edges,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )


def neighbor_graph_from_ijs(
    i: Array,
    j: Array,
    S: Array,
    coord: Array,
    box: Array | None,
    nframe_id: Array,
    nloc: int,
    layout: GraphLayout | None = None,
    *,
    with_csr: bool = False,
    canonicalize: bool = False,
) -> NeighborGraph:
    """Convert a sparse ``(i, j, S)`` edge list into a :class:`NeighborGraph`.

    ``edge_vec`` is recomputed from ``coord``/``box`` (NOT from any distance vector
    the search backend may carry), so it is a differentiable function of the input
    coordinates and follows the graph convention exactly.

    Parameters
    ----------
    i
        (E,) int per-edge center, per-frame LOCAL index in ``[0, nloc)``.
    j
        (E,) int per-edge neighbor, per-frame LOCAL index in ``[0, nloc)``.
    S
        (E, 3) int periodic-image shift: the neighbor sits at ``coord[j] + S @ box``.
    coord
        (nf, nloc, 3) local coordinates.
    box
        (nf, 3, 3) simulation cell, or ``None`` for non-periodic (``S`` ignored).
    nframe_id
        (E,) int frame index of each edge.
    nloc
        number of local atoms per frame (used for the frame-major node offset).
    layout
        edge-axis length policy; ``None`` => dynamic (torch) with ``min_edges`` guards.
    with_csr
        Whether to construct destination/source CSR views for a consumer that
        requires edge-grouped reductions.
    canonicalize
        Whether to reorder every edge field into destination-major form. Implies
        ``with_csr=True``.

    Returns
    -------
    NeighborGraph
        ``edge_index = [j + nframe_id*nloc, i + nframe_id*nloc]`` (src=neighbor,
        dst=center); ``edge_vec = coord[j] + S@box - coord[i]``; ``n_node`` is
        ``nloc`` per frame.
    """
    if layout is None:
        layout = GraphLayout()
    with_csr = with_csr or canonicalize
    xp = array_api_compat.array_namespace(coord)
    dev = array_api_compat.device(coord)
    nf = coord.shape[0]
    coord = xp.reshape(coord, (nf, nloc, 3))
    i = xp.astype(xp.asarray(i, device=dev), xp.int64)
    j = xp.astype(xp.asarray(j, device=dev), xp.int64)
    nframe_id = xp.astype(xp.asarray(nframe_id, device=dev), xp.int64)
    # flat frame-major node indices
    i_flat = i + nframe_id * nloc
    j_flat = j + nframe_id * nloc
    coord_flat = xp.reshape(coord, (nf * nloc, 3))
    r_i = xp.take(coord_flat, i_flat, axis=0)
    r_j = xp.take(coord_flat, j_flat, axis=0)
    edge_vec = r_j - r_i
    if box is not None:
        box = xp.asarray(box, device=dev)
        box = xp.reshape(box, (nf, 3, 3))
        box_per_edge = xp.take(box, nframe_id, axis=0)  # (E, 3, 3)
        S = xp.astype(xp.asarray(S, device=dev), box.dtype)
        # S @ box per edge via broadcast sum (NEVER np.einsum, which breaks on torch):
        # shift[e, b] = sum_a S[e, a] * box[e, a, b]
        shift = xp.sum(S[:, :, None] * box_per_edge, axis=1)  # (E, 3)
        edge_vec = edge_vec + shift
    edge_index = xp.stack([j_flat, i_flat], axis=0)
    edge_index, edge_vec, edge_mask = pad_and_guard_edges(
        edge_index, edge_vec, layout.edge_capacity, layout.min_edges
    )
    n_node = xp.full((nf,), nloc, dtype=xp.int64, device=dev)
    if not with_csr:
        return NeighborGraph(
            n_node=n_node,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
        )
    (
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_row_ptr,
        source_order,
    ) = build_edge_csr(
        edge_index,
        edge_vec,
        edge_mask,
        nf * nloc,
        canonicalize=canonicalize,
    )
    return NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
        destination_order=destination_order,
        destination_row_ptr=destination_row_ptr,
        source_row_ptr=source_row_ptr,
        source_order=source_order,
        destination_sorted=canonicalize,
    )
