# SPDX-License-Identifier: LGPL-3.0-or-later
"""Builders that produce a :class:`NeighborGraph`.

CONTRACT — these are the **legacy-compatible** builders: the graph they return
carries exactly the neighbors of the dense nlist they are built from, i.e. it
INHERITS that nlist's ``sel`` selection/truncation. They are the backward-compat
adapter (and the test oracle), NOT the graph-native "all neighbors within
``rcut``" builder. The complete-environment (carry-all, ``sel``-as-normalization)
contract is provided by a SEPARATE builder (``from_ijs`` fed by ASE/vesin; see
memory/spec_unified_edge_nlist.md decision #17). Keeping the two contracts in
distinct functions avoids the footgun of a consumer assuming it sees all
neighbors while a builder silently truncated them.

- ``neighbor_graph_from_extended``: convert an existing extended quartet
  (extended_coord, nlist, mapping) -> ghost-free graph (inherits the nlist's sel).
- ``build_neighbor_graph``: reuse deepmd's tested
  ``extend_input_and_build_neighbor_list`` (which TRUNCATES to ``sel``) then the
  adapter -> legacy-compatible graph.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import array_api_compat

from .graph import (
    GraphLayout,
    NeighborGraph,
    pad_and_guard_edges,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )


def neighbor_graph_from_extended(
    extended_coord: Array,
    nlist: Array,
    mapping: Array,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Convert the legacy extended quartet into a ghost-free NeighborGraph.

    This is the dpmodel/array-API adapter that REUSES deepmd's existing, tested,
    general-cell neighbor list (``build_neighbor_list`` / ``extend_coord_with_ghosts``)
    instead of re-deriving neighbors. For each valid neighbor slot it emits one
    edge with ``src = mapping[neighbor]`` (the neighbor's LOCAL owner -> ghost-free
    index), ``dst = center`` (local), and ``edge_vec = extended_coord[neighbor] -
    extended_coord[center]`` (the ghost coordinate already carries the periodic
    shift). Invalid slots (``nlist == -1``) are dropped. Nodes are flattened with a
    ``frame * nloc`` offset; the edge axis is padded/guarded via ``pad_and_guard_edges``.

    Because every neighbor maps to a LOCAL owner, the resulting graph is ghost-free:
    forces scatter to local atoms (periodic images of the same atom sum to one owner
    through the ``src`` index), so no ``edge_scatter_index`` is needed (single-rank).

    Parameters
    ----------
    extended_coord
        (nf, nall, 3) extended (local + ghost) coordinates.
    nlist
        (nf, nloc, nsel) neighbor list into the extended atoms; -1 is padding.
    mapping
        (nf, nall) extended -> local-owner index (local atoms map to themselves).
    layout
        edge-axis length policy; ``None`` => dynamic (torch) with ``min_edges`` guards.
    """
    if layout is None:
        layout = GraphLayout()
    xp = array_api_compat.array_namespace(extended_coord, nlist, mapping)
    dev = array_api_compat.device(extended_coord)
    nf = nlist.shape[0]
    nloc = nlist.shape[1]
    nsel = nlist.shape[2]
    n_node = xp.full((nf,), nloc, dtype=xp.int64, device=dev)
    src_parts: list[Array] = []
    dst_parts: list[Array] = []
    vec_parts: list[Array] = []
    center_full = xp.broadcast_to(
        xp.reshape(xp.arange(nloc, dtype=xp.int64, device=dev), (nloc, 1)),
        (nloc, nsel),
    )
    center_flat = xp.reshape(center_full, (nloc * nsel,))
    for ff in range(nf):
        nl_flat = xp.reshape(nlist[ff], (nloc * nsel,))
        keep = xp.reshape(xp.nonzero(nl_flat >= 0)[0], (-1,))
        j_ext = xp.take(nl_flat, keep, axis=0)  # extended neighbor indices
        dst = xp.take(center_flat, keep, axis=0)  # local center indices
        src = xp.take(mapping[ff], j_ext, axis=0)  # local owner of neighbor
        vec = xp.take(extended_coord[ff], j_ext, axis=0) - xp.take(
            extended_coord[ff], dst, axis=0
        )
        offset = ff * nloc
        src_parts.append(src + offset)
        dst_parts.append(dst + offset)
        vec_parts.append(vec)
    edge_index = xp.astype(
        xp.stack([xp.concat(src_parts), xp.concat(dst_parts)], axis=0), xp.int64
    )
    edge_vec = xp.concat(vec_parts, axis=0)
    edge_index, edge_vec, edge_mask = pad_and_guard_edges(
        edge_index, edge_vec, layout.edge_capacity, layout.min_edges
    )
    return NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
    )


def build_neighbor_graph(
    coord: Array,
    atype: Array,
    box: Array | None,
    rcut: float,
    sel: int | list[int],
    mixed_types: bool = True,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Build a LEGACY-COMPATIBLE NeighborGraph by reusing the tested dense nlist.

    Calls ``extend_input_and_build_neighbor_list`` (general-cell, tested) then
    :func:`neighbor_graph_from_extended`.

    CONTRACT: the returned graph contains the neighbors selected by ``sel`` and
    **inherits the legacy ``sel`` truncation** — it does NOT carry all neighbors
    within ``rcut`` when ``sel`` binds. It coincides with the complete in-``rcut``
    environment ONLY when ``sel`` is large enough that no real neighbor is dropped
    (the ``sel``-as-normalization regime). For the carry-all graph-native contract
    use the dedicated carry-all builder (``from_ijs`` via ASE/vesin), NOT this
    function. See memory/spec_unified_edge_nlist.md (decision #17). This builder is
    the backward-compat adapter and the test oracle.
    """
    from deepmd.dpmodel.utils.nlist import (
        extend_input_and_build_neighbor_list,
    )

    # ``extend_input_and_build_neighbor_list`` is annotated ``sel: list[int]``;
    # normalize the integer form so the public ``int | list[int]`` contract is
    # honored (the underlying ``build_neighbor_list`` accepts both).
    sel_list = [sel] if isinstance(sel, int) else sel
    extended_coord, _extended_atype, mapping, nlist = (
        extend_input_and_build_neighbor_list(
            coord, atype, rcut, sel_list, mixed_types=mixed_types, box=box
        )
    )
    return neighbor_graph_from_extended(extended_coord, nlist, mapping, layout)
