# SPDX-License-Identifier: LGPL-3.0-or-later
"""Builders/converters that produce a :class:`NeighborGraph`.

Two distinct groups (see memory/spec_unified_edge_nlist.md decision #17), kept
separate so a consumer can never assume completeness while a function silently
truncated:

1. **Dispatcher (compute from raw geometry).** ``build_neighbor_graph`` takes
   coordinates/box/types -- *no pre-existing list* -- and SEARCHES for neighbors,
   returning a CARRY-ALL graph: every neighbor within ``rcut``. ``sel`` is
   normalization-only (consumed downstream by the descriptor) and is NEVER a
   cutoff here. This module ships the ``dense`` (all-pairs, O(N^2) reference)
   search; O(N) ``vesin``/``ase`` backends land later behind a ``method`` key.

2. **Converters (adapt an already-built list).** ``from_dense_quartet`` adapts an
   existing extended quartet (extended_coord, nlist, mapping) into a graph. It
   performs NO search and therefore INHERITS that quartet's ``sel`` truncation --
   it is the backward-compat bridge to the legacy dense nlist (World 1) and the
   test oracle, NOT a carry-all path. The ``(i,j,S)`` converter (``from_ijs``,
   fed by ASE/vesin/LAMMPS) lands with the dispatcher's O(N) backends.

The dispatcher and the converters share the format-conversion code (a search
backend = search + its converter as the final step); they are separate only on
the question "did I get raw geometry, or an already-built list?".
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


def from_dense_quartet(
    extended_coord: Array,
    nlist: Array,
    mapping: Array,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Convert a legacy extended quartet into a ghost-free NeighborGraph (CONVERTER).

    This is a backward-compat CONVERTER (World 1 -> graph): it performs NO neighbor
    search and INHERITS the ``sel`` truncation already baked into ``nlist``. Use it
    only when a caller (an MD code, or the legacy dense path) already holds a
    built quartet; for the carry-all graph use :func:`build_neighbor_graph`.

    For each valid neighbor slot it emits one edge with ``src = mapping[neighbor]``
    (the neighbor's LOCAL owner -> ghost-free index), ``dst = center`` (local), and
    ``edge_vec = extended_coord[neighbor] - extended_coord[center]`` (the ghost
    coordinate already carries the periodic shift). Invalid slots (``nlist == -1``)
    are dropped. Nodes are flattened with a ``frame * nloc`` offset; the edge axis
    is padded/guarded via ``pad_and_guard_edges``.

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
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Build a CARRY-ALL NeighborGraph DIRECTLY from coordinates (``dense`` search).

    This is the dispatcher's reference ``dense`` backend: it SEARCHES for neighbors
    from raw geometry and emits EVERY neighbor within ``rcut``. It is **sel-free** --
    there is intentionally no ``sel`` parameter, because ``sel`` is normalization-only
    (consumed by the descriptor downstream) and never a cutoff. It does NOT route
    through the legacy dense nlist / :func:`from_dense_quartet`, so it carries no
    ``sel`` truncation.

    Implementation: reuse the tested periodic ghosting
    (:func:`~deepmd.dpmodel.utils.nlist.extend_coord_with_ghosts`) to materialise all
    periodic images within ``rcut``, then enumerate all center-neighbor pairs within
    ``rcut`` UNCAPPED. This is an O(N^2) reference search (correctness oracle); the
    O(N) ``vesin``/``ase`` backends arrive later behind a ``method`` key. Edges map
    every neighbor to its LOCAL owner (``src = mapping[neighbor]``), so the graph is
    ghost-free.

    Parameters
    ----------
    coord
        (nf, nloc, 3) or (nf, nloc*3) local coordinates.
    atype
        (nf, nloc) local atom types; ``type < 0`` marks a virtual atom (excluded
        as both a center and a neighbor).
    box
        (nf, 3, 3) or (nf, 9) simulation cell; ``None`` for non-periodic.
    rcut
        cutoff radius (neighbors kept where ``0 < |edge_vec| <= rcut``, matching the
        legacy nlist convention so this coincides with :func:`from_dense_quartet`
        at non-binding ``sel``).
    layout
        edge-axis length policy; ``None`` => dynamic (torch) with ``min_edges`` guards.
    """
    from deepmd.dpmodel.utils.nlist import (
        extend_coord_with_ghosts,
    )
    from deepmd.dpmodel.utils.region import (
        normalize_coord,
    )

    if layout is None:
        layout = GraphLayout()
    xp = array_api_compat.array_namespace(coord, atype)
    dev = array_api_compat.device(coord)
    nf, nloc = atype.shape[:2]
    coord = xp.reshape(coord, (nf, nloc, 3))
    if box is not None:
        box = xp.reshape(box, (nf, 3, 3))
        coord = normalize_coord(coord, box)
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord, atype, box, rcut
    )
    extended_coord = xp.reshape(extended_coord, (nf, -1, 3))
    nall = extended_coord.shape[1]
    n_node = xp.full((nf,), nloc, dtype=xp.int64, device=dev)
    arange_nloc = xp.arange(nloc, dtype=xp.int64, device=dev)
    arange_nall = xp.arange(nall, dtype=xp.int64, device=dev)
    # (nloc, nall) static index grids reused per frame
    ii_flat = xp.reshape(
        xp.broadcast_to(arange_nloc[:, None], (nloc, nall)), (nloc * nall,)
    )
    jj_flat = xp.reshape(
        xp.broadcast_to(arange_nall[None, :], (nloc, nall)), (nloc * nall,)
    )
    src_parts: list[Array] = []
    dst_parts: list[Array] = []
    vec_parts: list[Array] = []
    for ff in range(nf):
        ec = extended_coord[ff]  # (nall, 3)
        centers = ec[:nloc, :]  # (nloc, 3)
        diff = (
            ec[None, :, :] - centers[:, None, :]
        )  # (nloc, nall, 3) = ext[j]-center[i]
        dist = xp.linalg.vector_norm(diff, axis=-1)  # (nloc, nall)
        # keep neighbors within rcut, dropping: the self extended atom (i==j;
        # a periodic IMAGE of i has j!=i and is kept), virtual neighbors, and
        # virtual centers. Uncapped -- no sel truncation.
        not_self = jj_flat != ii_flat  # (nloc*nall,)
        vir_nei = xp.reshape(
            xp.broadcast_to((extended_atype[ff] < 0)[None, :], (nloc, nall)),
            (nloc * nall,),
        )
        vir_cen = xp.reshape(
            xp.broadcast_to((atype[ff] < 0)[:, None], (nloc, nall)),
            (nloc * nall,),
        )
        within = xp.reshape(dist <= rcut, (nloc * nall,))
        keep_mask = (
            within & not_self & xp.logical_not(vir_nei) & xp.logical_not(vir_cen)
        )
        keep = xp.reshape(xp.nonzero(keep_mask)[0], (-1,))
        dst = xp.take(ii_flat, keep, axis=0)  # local center
        j_ext = xp.take(jj_flat, keep, axis=0)  # extended neighbor index
        src = xp.take(mapping[ff], j_ext, axis=0)  # local owner of neighbor
        vec = xp.take(xp.reshape(diff, (nloc * nall, 3)), keep, axis=0)
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
