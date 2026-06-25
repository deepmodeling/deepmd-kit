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
)

import numpy as np

from .from_ijs import (
    neighbor_graph_from_ijs,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )

    from .graph import (
        GraphLayout,
        NeighborGraph,
    )


def build_neighbor_graph_ase(
    coord: Array,
    atype: Array,
    box: Array | None,
    rcut: float,
    layout: GraphLayout | None = None,
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
        (nf, nloc) local atom types (unused for the search; carried for API parity).
    box
        (nf, 3, 3) simulation cell, or ``None`` for non-periodic.
    rcut
        cutoff radius.
    layout
        edge-axis length policy; ``None`` => dynamic (torch) with ``min_edges`` guards.

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

    coord_np = np.asarray(coord)
    nf, nloc = coord_np.shape[:2]
    coord_np = coord_np.reshape(nf, nloc, 3)
    box_np = np.asarray(box).reshape(nf, 3, 3) if box is not None else None
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

    return neighbor_graph_from_ijs(
        i_all, j_all, S_all, coord, box, nframe_all, nloc, layout=layout
    )
