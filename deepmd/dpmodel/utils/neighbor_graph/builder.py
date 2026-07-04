# SPDX-License-Identifier: LGPL-3.0-or-later
"""Builders/converters that produce a :class:`NeighborGraph`.

Two distinct groups (see the design discussion wanghan-iapcm/deepmd-kit#4 decision #17), kept
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

Both are fully vectorized over the frame axis (no Python frame loop): per-slot
``(frame, center, neighbor)`` index grids are flattened, masked, and gathered in
one shot, with cross-frame gathers done through ``frame * nall + idx`` flat
indices.
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


def from_dense_quartet(
    extended_coord: Array,
    nlist: Array,
    mapping: Array,
    layout: GraphLayout | None = None,
    compact: bool = True,
    *,
    with_csr: bool = False,
    canonicalize: bool = False,
) -> NeighborGraph:
    """Convert a legacy extended quartet into a ghost-free NeighborGraph (CONVERTER).

    This is a backward-compat CONVERTER (World 1 -> graph): it performs NO neighbor
    search and INHERITS the ``sel`` truncation already baked into ``nlist``. Use it
    only when a caller (an MD code, or the legacy dense path) already holds a
    built quartet. In contrast, the carry-all graph builders search from RAW
    coordinates and apply NO ``sel`` truncation: :func:`build_neighbor_graph`
    (the ``neighbor_graph_method="dense"`` all-pairs route) and
    :func:`build_neighbor_graph_ase` (the ``"ase"`` O(N) cell-list route).

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
    compact
        If True (default), COMPACT real edges with ``nonzero`` and pad/guard via
        :func:`pad_and_guard_edges` -- the data-dependent output shape breaks
        jax.jit / torch.export. If False, emit a SHAPE-STATIC graph: every nlist
        slot becomes an edge (``E = nf * nloc * nsel``, a static shape), invalid
        slots (``nlist == -1``) get ``edge_mask=False``, zero ``edge_vec`` and a
        ``src`` pointing at the center (in-range, masked) -- so no ``nonzero`` is
        used and the converter is jit/export-traceable. The masked edges contribute
        zero in a downstream ``segment_sum``, so the descriptor output is unchanged.
    with_csr
        Whether to construct destination/source CSR views for a consumer that
        requires edge-grouped reductions.
    canonicalize
        Whether to reorder every edge field into destination-major form. Implies
        ``with_csr=True``.

    Returns
    -------
    graph
        The :class:`NeighborGraph` over the LOCAL atoms (``n_node = nloc`` per
        frame): ``edge_index`` ``[src, dst]`` in local indices, ``edge_vec`` the
        neighbor-minus-center displacement, and ``edge_mask`` flagging real edges.
    """
    if layout is None:
        layout = GraphLayout()
    with_csr = with_csr or canonicalize
    xp = array_api_compat.array_namespace(extended_coord, nlist, mapping)
    dev = array_api_compat.device(extended_coord)
    nf, nloc, nsel = nlist.shape
    nall = extended_coord.shape[1]
    if not compact:
        if layout.edge_capacity is not None:
            raise NotImplementedError(
                "shape-static from_dense_quartet pads to E=nf*nloc*nsel; "
                "edge_capacity unsupported here"
            )
        # (E,) flat grids, E = nf*nloc*nsel, row-major (frame, center, slot)
        ff = xp.reshape(
            xp.broadcast_to(
                xp.reshape(xp.arange(nf, dtype=xp.int64, device=dev), (nf, 1, 1)),
                (nf, nloc, nsel),
            ),
            (-1,),
        )
        center = xp.reshape(
            xp.broadcast_to(
                xp.reshape(xp.arange(nloc, dtype=xp.int64, device=dev), (1, nloc, 1)),
                (nf, nloc, nsel),
            ),
            (-1,),
        )
        nl = xp.reshape(nlist, (-1,))  # neighbor ext idx or -1
        valid = nl >= 0  # (E,) bool <-- the mask
        j_safe = xp.where(valid, nl, xp.zeros_like(nl))  # clamp -1 -> 0 (avoid OOB)
        ec_flat = xp.reshape(extended_coord, (nf * nall, 3))
        map_flat = xp.reshape(mapping, (nf * nall,))
        g_nei = ff * nall + j_safe
        g_cen = ff * nall + center
        src_local = xp.take(map_flat, g_nei, axis=0)
        edge_vec = xp.take(ec_flat, g_nei, axis=0) - xp.take(ec_flat, g_cen, axis=0)
        edge_vec = edge_vec * xp.astype(valid[:, None], edge_vec.dtype)  # zero invalid
        src = xp.where(valid, ff * nloc + src_local, ff * nloc + center)  # -> center
        dst = ff * nloc + center
        edge_index = xp.astype(xp.stack([src, dst], axis=0), xp.int64)
        edge_mask = valid
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
    else:
        # COMPACT: drop invalid slots via nonzero (dynamic shape -> eager only,
        # NOT jit/export-traceable) then pad/guard.
        # per-slot (nf, nloc, nsel) index grids, flattened frame-major
        ff_grid = xp.broadcast_to(
            xp.reshape(xp.arange(nf, dtype=xp.int64, device=dev), (nf, 1, 1)),
            (nf, nloc, nsel),
        )
        center_grid = xp.broadcast_to(
            xp.reshape(xp.arange(nloc, dtype=xp.int64, device=dev), (1, nloc, 1)),
            (nf, nloc, nsel),
        )
        ff_flat = xp.reshape(ff_grid, (-1,))
        center_flat = xp.reshape(center_grid, (-1,))
        nl_flat = xp.reshape(nlist, (-1,))
        keep = xp.reshape(xp.nonzero(nl_flat >= 0)[0], (-1,))
        ff_k = xp.take(ff_flat, keep, axis=0)
        dst_local = xp.take(center_flat, keep, axis=0)  # center index in [0, nloc)
        j_ext = xp.take(nl_flat, keep, axis=0)  # neighbor index in [0, nall)
        # cross-frame gathers via flat (frame * nall + idx) indices; centers are
        # the first nloc extended atoms (local atoms precede ghosts).
        ec_flat = xp.reshape(extended_coord, (nf * nall, 3))
        map_flat = xp.reshape(mapping, (nf * nall,))
        g_nei = ff_k * nall + j_ext
        g_cen = ff_k * nall + dst_local
        src_local = xp.take(map_flat, g_nei, axis=0)  # local owner of the neighbor
        edge_vec = xp.take(ec_flat, g_nei, axis=0) - xp.take(ec_flat, g_cen, axis=0)
        edge_index = xp.astype(
            xp.stack([ff_k * nloc + src_local, ff_k * nloc + dst_local], axis=0),
            xp.int64,
        )
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


def build_neighbor_graph(
    coord: Array,
    atype: Array,
    box: Array | None,
    rcut: float,
    layout: GraphLayout | None = None,
    *,
    with_csr: bool = False,
    canonicalize: bool = False,
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
    ``rcut`` UNCAPPED, vectorized over frames. This is an O(N^2) reference search
    (correctness oracle); the O(N) ``vesin``/``ase`` backends arrive later behind a
    ``method`` key. Edges map every neighbor to its LOCAL owner
    (``src = mapping[neighbor]``), so the graph is ghost-free.

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
    with_csr
        Whether to construct destination/source CSR views for a consumer that
        requires edge-grouped reductions.
    canonicalize
        Whether to reorder every edge field into destination-major form. Implies
        ``with_csr=True``.
    """
    from deepmd.dpmodel.utils.nlist import (
        extend_coord_with_ghosts,
    )
    from deepmd.dpmodel.utils.region import (
        normalize_coord,
    )

    if layout is None:
        layout = GraphLayout()
    with_csr = with_csr or canonicalize
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
    # all center-neighbor displacements: (nf, nloc, nall, 3) = ext[j] - center[i]
    centers = extended_coord[:, :nloc, :]
    diff = extended_coord[:, None, :, :] - centers[:, :, None, :]
    dist = xp.linalg.vector_norm(diff, axis=-1)  # (nf, nloc, nall)
    # per-slot (nf, nloc, nall) index grids
    ff_grid = xp.broadcast_to(
        xp.reshape(xp.arange(nf, dtype=xp.int64, device=dev), (nf, 1, 1)),
        (nf, nloc, nall),
    )
    i_grid = xp.broadcast_to(
        xp.reshape(xp.arange(nloc, dtype=xp.int64, device=dev), (1, nloc, 1)),
        (nf, nloc, nall),
    )
    j_grid = xp.broadcast_to(
        xp.reshape(xp.arange(nall, dtype=xp.int64, device=dev), (1, 1, nall)),
        (nf, nloc, nall),
    )
    # keep neighbors within rcut, dropping: the self extended atom (i==j; a periodic
    # IMAGE of i has j!=i and is kept), virtual neighbors, and virtual centers.
    not_self = j_grid != i_grid
    vir_nei = xp.broadcast_to((extended_atype < 0)[:, None, :], (nf, nloc, nall))
    vir_cen = xp.broadcast_to((atype < 0)[:, :, None], (nf, nloc, nall))
    keep_mask = (
        (dist <= rcut) & not_self & xp.logical_not(vir_nei) & xp.logical_not(vir_cen)
    )
    keep = xp.reshape(xp.nonzero(xp.reshape(keep_mask, (-1,)))[0], (-1,))
    ff_k = xp.take(xp.reshape(ff_grid, (-1,)), keep, axis=0)
    dst_local = xp.take(xp.reshape(i_grid, (-1,)), keep, axis=0)  # local center
    j_ext = xp.take(xp.reshape(j_grid, (-1,)), keep, axis=0)  # extended neighbor
    edge_vec = xp.take(xp.reshape(diff, (nf * nloc * nall, 3)), keep, axis=0)
    # cross-frame neighbor-owner gather via flat (frame * nall + idx)
    map_flat = xp.reshape(mapping, (nf * nall,))
    src_local = xp.take(map_flat, ff_k * nall + j_ext, axis=0)
    edge_index = xp.astype(
        xp.stack([ff_k * nloc + src_local, ff_k * nloc + dst_local], axis=0), xp.int64
    )
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
