# SPDX-License-Identifier: LGPL-3.0-or-later
"""Carry-all neighbor graphs from the NumPy interface of Vesin's cell list."""

from itertools import (
    pairwise,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np

from .from_ijs import (
    neighbor_graph_from_ijs,
)
from .graph import (
    GraphLayout,
    NeighborGraph,
    apply_pair_exclusion,
    pad_and_guard_edges,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )


def build_neighbor_graph_vesin(
    coord: np.ndarray,
    atype: np.ndarray,
    box: np.ndarray | None,
    rcut: float,
    layout: GraphLayout | None = None,
    *,
    n_node: np.ndarray | None = None,
    pair_excl: "PairExcludeMask | None" = None,
) -> NeighborGraph:
    """Enumerate every cutoff neighbor on a flat or rectangular node axis.

    Vesin searches on the host without a dense all-pairs distance matrix.
    Its topology passes through the same ``(i, j, S)`` converter as the ASE
    builder, preserving periodic images and the neighbor-minus-center vector
    convention. ``sel`` does not participate in the search.

    Parameters
    ----------
    coord : numpy.ndarray
        Coordinates with shape (N, 3) or (nf, nloc, 3), in Angstrom.
    atype : numpy.ndarray
        Atom types with shape (N,) or (nf, nloc). Negative types are padding.
    box : numpy.ndarray or None
        Periodic cells with shape (nf, 3, 3), in Angstrom; None is nonperiodic.
    rcut : float
        Cutoff radius, in Angstrom.
    layout : GraphLayout or None
        Edge capacity and guard policy.
    n_node : numpy.ndarray or None
        Frame node counts with shape (nf,), required for a flat batch.
    pair_excl : PairExcludeMask or None
        Model-level pair exclusions, applied before edge padding.

    Returns
    -------
    NeighborGraph
        A graph containing all valid directed neighbors within the cutoff.
    """
    from vesin import (
        NeighborList,
    )

    if n_node is None:
        if atype.ndim != 2:
            raise ValueError("A flat neighbor-graph batch requires n_node.")
        n_node = np.full(atype.shape[0], atype.shape[1], dtype=np.int64)
    coord = np.asarray(coord).reshape(-1, 3)
    atype = np.asarray(atype).reshape(-1)
    n_node = np.asarray(n_node, dtype=np.int64)
    cells = None if box is None else np.asarray(box).reshape(-1, 3, 3)
    offsets = np.concatenate(([0], np.cumsum(n_node)))
    search = NeighborList(cutoff=rcut, full_list=True, sorted=True)
    centers, neighbors, shifts, frames = [], [], [], []
    for frame, (start, end) in enumerate(pairwise(offsets)):
        keep = np.flatnonzero(atype[start:end] >= 0)
        if not keep.size:
            continue
        cell = np.eye(3, dtype=np.float64) if cells is None else cells[frame]
        i, j, image = search.compute(
            np.asarray(coord[start:end][keep], dtype=np.float64),
            np.asarray(cell, dtype=np.float64),
            periodic=cells is not None,
            quantities="ijS",
        )
        centers.append(keep[i])
        neighbors.append(keep[j])
        shifts.append(image)
        frames.append(np.full(i.shape, frame, dtype=np.int64))
    i = np.concatenate(centers) if centers else np.empty(0, dtype=np.int64)
    j = np.concatenate(neighbors) if neighbors else np.empty(0, dtype=np.int64)
    image = np.concatenate(shifts) if shifts else np.empty((0, 3), dtype=np.int64)
    frame_id = np.concatenate(frames) if frames else np.empty(0, dtype=np.int64)
    graph = neighbor_graph_from_ijs(
        i,
        j,
        image,
        coord,
        cells,
        frame_id,
        n_node,
        layout=GraphLayout(min_edges=0),
    )
    if pair_excl is not None:
        graph = apply_pair_exclusion(graph, atype, pair_excl, compact=True)
    if layout is None:
        layout = GraphLayout()
    graph.edge_index, graph.edge_vec, graph.edge_mask = pad_and_guard_edges(
        graph.edge_index, graph.edge_vec, layout.edge_capacity, layout.min_edges
    )
    return graph
