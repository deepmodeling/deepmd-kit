# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
)
from deepmd.dpmodel.utils.numpy_neighbor_list import (
    NumpyNeighborList,
)


def brute_force_neighbor_sets(coord, box, rcut):
    """Reference: per center i, the multiset of (src j, rounded edge_vec)."""
    nloc = coord.shape[0]
    if box is None:
        shells = [np.zeros(3, dtype=np.int64)]
    else:
        h = np.min(np.abs(np.diag(box)))
        n = int(np.ceil(rcut / h))
        shells = [
            np.array(s, dtype=np.int64)
            for s in itertools.product(range(-n, n + 1), repeat=3)
        ]
    sets = [set() for _ in range(nloc)]
    for s in shells:
        sc = np.zeros(3) if box is None else s.astype(float) @ box
        for i in range(nloc):
            for j in range(nloc):
                vec = coord[j] + sc - coord[i]
                r = np.linalg.norm(vec)
                if 1e-10 < r < rcut:
                    sets[i].add((j, tuple(np.round(vec, 6))))
    return sets


class TestNumpyNeighborList(unittest.TestCase):
    def setUp(self) -> None:
        self.rcut = 4.0
        self.coord = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.5, 0.0, 0.0]],
            dtype=np.float64,
        ).reshape(1, 4, 3)

    def _graph_neighbor_sets(self, ng, nloc):
        ei = ng.edge_index[:, ng.edge_mask]  # real edges only
        ev = ng.edge_vec[ng.edge_mask]
        sets = [set() for _ in range(nloc)]
        for k in range(ei.shape[1]):
            src, dst = int(ei[0, k]), int(ei[1, k])
            sets[dst].add((src, tuple(np.round(ev[k], 6))))
        return sets

    def test_nonperiodic_matches_brute_force(self) -> None:
        ng = NumpyNeighborList().build(self.coord, box=None, rcut=self.rcut)
        np.testing.assert_array_equal(ng.n_node, np.array([4], dtype=np.int64))
        got = self._graph_neighbor_sets(ng, 4)
        want = brute_force_neighbor_sets(self.coord[0], None, self.rcut)
        self.assertEqual(got, want)

    def test_periodic_matches_brute_force(self) -> None:
        box = np.eye(3, dtype=np.float64)[None] * 6.0  # (1,3,3) cubic L=6
        ng = NumpyNeighborList().build(self.coord, box=box, rcut=self.rcut)
        got = self._graph_neighbor_sets(ng, 4)
        want = brute_force_neighbor_sets(self.coord[0], box[0], self.rcut)
        self.assertEqual(got, want)

    def test_edge_vec_within_rcut(self) -> None:
        ng = NumpyNeighborList().build(self.coord, box=None, rcut=self.rcut)
        ev = ng.edge_vec[ng.edge_mask]
        self.assertTrue(np.all(np.linalg.norm(ev, axis=1) < self.rcut))

    def test_multiframe_offsets_nodes(self) -> None:
        coord2 = np.concatenate([self.coord, self.coord], axis=0)  # nf=2, nloc=4
        ng = NumpyNeighborList().build(coord2, box=None, rcut=self.rcut)
        np.testing.assert_array_equal(ng.n_node, np.array([4, 4], dtype=np.int64))
        ei = ng.edge_index[:, ng.edge_mask]
        # every frame-1 edge endpoint is in [4, 8); every frame-0 edge in [0, 4)
        f0 = ei[:, ei[1] < 4]
        f1 = ei[:, ei[1] >= 4]
        self.assertTrue(np.all(f0 < 4))
        self.assertTrue(np.all(f1 >= 4))

    def test_static_capacity_padding(self) -> None:
        ng = NumpyNeighborList().build(
            self.coord, box=None, rcut=self.rcut, layout=GraphLayout(edge_capacity=64)
        )
        self.assertEqual(ng.edge_index.shape[1], 64)
        self.assertEqual(ng.edge_vec.shape[0], 64)
        self.assertEqual(int(ng.edge_mask.sum()), int(ng.edge_mask[:].sum()))
        # masked-out tail contributes no real edges
        self.assertTrue(np.all(ng.edge_vec[~ng.edge_mask] == 0.0))
