# SPDX-License-Identifier: LGPL-3.0-or-later
"""Carry-all NeighborGraph builder backed by ASE's O(N) cell list (optional dep).

``build_neighbor_graph_ase`` is a carry-all search backend: it uses ASE's
``neighbor_list("ijS", ...)`` to enumerate EVERY neighbor within ``rcut`` (no
``sel`` cutoff), then routes the resulting sparse ``(i, j, S)`` edge list through
:func:`neighbor_graph_from_ijs` so ``edge_vec`` is recomputed differentiably from
``coord``/``box`` -- ASE's own distance vectors are intentionally NOT used, to
keep the geometry convention and autograd leaf consistent with every other
builder. ASE is an OPTIONAL dependency, imported lazily inside the function.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from .csr import (
    attach_edge_csr,
)
from .from_ijs import (
    neighbor_graph_from_ijs,
)
from .graph import (
    apply_pair_exclusion,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )

    from .graph import (
        GraphLayout,
        NeighborGraph,
        PairExcludeMask,
    )


def build_neighbor_graph_ase(
    coord: Array,
    atype: Array,
    box: Array | None,
    rcut: float,
    layout: GraphLayout | None = None,
    *,
    with_csr: bool = False,
    canonicalize: bool = False,
    pair_excl: PairExcludeMask | None = None,
    compact: bool = False,
) -> NeighborGraph:
    """Build a CARRY-ALL NeighborGraph using ASE's O(N) cell-list search.

    Per frame, ASE ``neighbor_list("ijS", atoms, rcut)`` returns center ``i``,
    neighbor ``j`` and periodic shift ``S`` such that the neighbor image sits at
    ``positions[j] + S @ cell``. These map directly to the graph convention
    (src=neighbor=j, dst=center=i), and the edge list is fed to
    :func:`neighbor_graph_from_ijs` which recomputes ``edge_vec`` from
    ``coord``/``box`` (ASE's distance vectors are discarded for convention +
    differentiability consistency).

    Parameters
    ----------
    coord
        (nf, nloc, 3) local coordinates.
    atype
        (nf, nloc) local atom types; ``type < 0`` marks a virtual atom, excluded
        as center and neighbor (the search itself is type-blind).
    box
        (nf, 3, 3) simulation cell, or ``None`` for non-periodic.
    rcut
        cutoff radius.
    layout
        edge-axis length policy; ``None`` => dynamic (torch) with ``min_edges`` guards.
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
        The carry-all :class:`NeighborGraph` over the LOCAL atoms
        (``n_node = nloc`` per frame), with ``edge_vec`` recomputed
        differentiably from ``coord``/``box``.

    Raises
    ------
    ImportError
        if the optional ``ase`` package is not installed.
    """
    try:
        from ase import (
            Atoms,
        )
        from ase.neighborlist import (
            neighbor_list,
        )
    except ImportError as e:
        raise ImportError(
            "build_neighbor_graph_ase requires the optional 'ase' package; "
            "install ase or use neighbor-graph method 'dense'."
        ) from e

    # The ASE topology search runs on the CPU in numpy; convert safely from a
    # CUDA / grad-requiring torch tensor (the original coord/box are still
    # passed to neighbor_graph_from_ijs below, which recomputes edge_vec
    # differentiably on the native backend/device).
    def _to_cpu_numpy(x: Any) -> np.ndarray:
        return np.asarray(x.detach().cpu()) if hasattr(x, "detach") else np.asarray(x)

    coord_np = _to_cpu_numpy(coord)
    nf, nloc = coord_np.shape[:2]
    coord_np = coord_np.reshape(nf, nloc, 3)
    box_np = _to_cpu_numpy(box).reshape(nf, 3, 3) if box is not None else None
    periodic = box is not None

    i_parts = []
    j_parts = []
    S_parts = []
    nframe_parts = []
    for f in range(nf):
        atoms = Atoms(
            positions=coord_np[f],
            cell=(box_np[f] if periodic else None),
            pbc=periodic,
        )
        ii, jj, SS = neighbor_list("ijS", atoms, rcut)
        i_parts.append(np.asarray(ii, dtype=np.int64))
        j_parts.append(np.asarray(jj, dtype=np.int64))
        S_parts.append(np.asarray(SS, dtype=np.int64).reshape(-1, 3))
        nframe_parts.append(np.full((len(ii),), f, dtype=np.int64))

    i_all = np.concatenate(i_parts) if i_parts else np.zeros((0,), dtype=np.int64)
    j_all = np.concatenate(j_parts) if j_parts else np.zeros((0,), dtype=np.int64)
    S_all = np.concatenate(S_parts) if S_parts else np.zeros((0, 3), dtype=np.int64)
    nframe_all = (
        np.concatenate(nframe_parts) if nframe_parts else np.zeros((0,), dtype=np.int64)
    )

    # virtual atoms (atype < 0) are excluded as centers AND neighbors -- the
    # World-2 builder contract shared with the dense reference builder; the
    # geometric search above cannot know about them.
    atype_np = _to_cpu_numpy(atype).reshape(nf, nloc)
    keep = (atype_np[nframe_all, i_all] >= 0) & (atype_np[nframe_all, j_all] >= 0)
    i_all, j_all = i_all[keep], j_all[keep]
    S_all, nframe_all = S_all[keep], nframe_all[keep]

    graph = neighbor_graph_from_ijs(
        i_all,
        j_all,
        S_all,
        coord,
        box,
        nframe_all,
        nloc,
        layout=layout,
    )
    if pair_excl is not None:
        import array_api_compat

        xp = array_api_compat.array_namespace(coord)
        atype_flat = xp.reshape(xp.asarray(atype), (-1,))
        graph = apply_pair_exclusion(graph, atype_flat, pair_excl, compact=compact)
    if with_csr or canonicalize:
        graph = attach_edge_csr(graph, nf * nloc, canonicalize=canonicalize)
    return graph
