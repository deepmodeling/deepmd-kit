# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numpy reference neighbor-list builder producing a NeighborGraph.

All-pairs over periodic-image shells (over-enumeration filtered by rcut) -
correct but not optimized; the production O(N) builders (vesin / nvalchemiops)
live in the pt/pt_expt backends (later plan). Reference/test use only.
Limitation: shell count from the box diagonal (orthorhombic-ish test boxes).
"""

import itertools

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
    pad_and_guard_edges,
)


def _frame_edges(pos: np.ndarray, box: np.ndarray | None, rcut: float):
    """Return (src_j, dst_i, edge_vec) for one frame. src=neighbor, dst=center."""
    nloc = pos.shape[0]
    if box is None:
        shells = [np.zeros(3, dtype=np.int64)]
    else:
        h = float(np.min(np.abs(np.diag(box))))
        n = int(np.ceil(rcut / h))
        shells = [
            np.array(s, dtype=np.int64)
            for s in itertools.product(range(-n, n + 1), repeat=3)
        ]
    src, dst, vec = [], [], []
    rcut2 = rcut * rcut
    for s in shells:
        sc = np.zeros(3) if box is None else s.astype(np.float64) @ box
        shifted = pos + sc  # (nloc, 3) positions of image-shifted neighbors
        for i in range(nloc):
            d = shifted - pos[i]  # (nloc, 3)
            r2 = np.sum(d * d, axis=1)
            for j in range(nloc):
                if 1e-20 < r2[j] < rcut2:
                    src.append(j)
                    dst.append(i)
                    vec.append(d[j])
    if len(src) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
        )
    return (
        np.array(src, dtype=np.int64),
        np.array(dst, dtype=np.int64),
        np.stack(vec, axis=0).astype(np.float64),
    )


class NumpyNeighborList:
    """Reference builder: coords -> NeighborGraph."""

    def build(
        self,
        coord: np.ndarray,
        box: np.ndarray | None,
        rcut: float,
        layout: GraphLayout | None = None,
    ) -> NeighborGraph:
        if layout is None:
            layout = GraphLayout()
        coord = np.asarray(coord, dtype=np.float64)
        nf, nloc = coord.shape[0], coord.shape[1]
        n_node = np.full((nf,), nloc, dtype=np.int64)
        src_all, dst_all, vec_all = [], [], []
        for f in range(nf):
            bx = None if box is None else np.asarray(box, dtype=np.float64)[f]
            src, dst, vec = _frame_edges(coord[f], bx, rcut)
            offset = f * nloc
            src_all.append(src + offset)
            dst_all.append(dst + offset)
            vec_all.append(vec)
        src_cat = np.concatenate(src_all) if src_all else np.zeros((0,), np.int64)
        dst_cat = np.concatenate(dst_all) if dst_all else np.zeros((0,), np.int64)
        edge_index = np.stack([src_cat, dst_cat], axis=0).astype(np.int64)  # (2, E)
        edge_vec = (
            np.concatenate(vec_all, axis=0)
            if vec_all
            else np.zeros((0, 3), np.float64)
        )
        edge_index, edge_vec, edge_mask = pad_and_guard_edges(
            edge_index, edge_vec, layout.edge_capacity, layout.min_edges
        )
        return NeighborGraph(
            n_node=n_node,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
        )
