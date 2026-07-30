# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sparse ``(i, j, S)`` edge-list converter to :class:`NeighborGraph`.

``neighbor_graph_from_ijs`` is the canonical sparse converter: it takes an
already-built sparse edge list -- per-edge center ``i``, neighbor ``j`` (both
indices within their own frame) and integer periodic-image shift ``S``
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
    n_node: Array,
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
        (E,) int per-edge center, index within its own frame.
    j
        (E,) int per-edge neighbor, index within its own frame.
    S
        (E, 3) int periodic-image shift: the neighbor sits at ``coord[j] + S @ box``.
    coord
        (N, 3) local coordinates, frame-major over ``n_node``.
    box
        (nf, 3, 3) simulation cell, or ``None`` for non-periodic (``S`` ignored).
    nframe_id
        (E,) int frame index of each edge.
    n_node
        (nf,) int atoms per frame. Frames occupy contiguous blocks of the node
        axis in order, so the prefix sums of this vector are the frame offsets
        that turn a within-frame index into a node index. A batch padded to a
        common width is the special case where every entry is that width.
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
        ``edge_index`` holds node indices (src=neighbor, dst=center) and
        ``edge_vec = coord[j] + S@box - coord[i]``.
    """
    if layout is None:
        layout = GraphLayout()
    with_csr = with_csr or canonicalize
    xp = array_api_compat.array_namespace(coord)
    dev = array_api_compat.device(coord)
    n_node = xp.astype(xp.asarray(n_node, device=dev), xp.int64)
    nf = n_node.shape[0]
    coord_flat = xp.reshape(coord, (-1, 3))
    i = xp.astype(xp.asarray(i, device=dev), xp.int64)
    j = xp.astype(xp.asarray(j, device=dev), xp.int64)
    nframe_id = xp.astype(xp.asarray(nframe_id, device=dev), xp.int64)
    # Within-frame indices become node indices through the frame offsets.
    offset = xp.take(xp.cumulative_sum(n_node) - n_node, nframe_id, axis=0)
    i_flat = i + offset
    j_flat = j + offset
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
        source_order,
        source_row_ptr,
    ) = build_edge_csr(
        edge_index,
        edge_vec,
        edge_mask,
        int(coord_flat.shape[0]),
        canonicalize=canonicalize,
    )
    return NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
        destination_order=destination_order,
        destination_row_ptr=destination_row_ptr,
        source_order=source_order,
        source_row_ptr=source_row_ptr,
        destination_sorted=canonicalize,
    )
