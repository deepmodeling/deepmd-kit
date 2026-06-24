# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    edge_force_virial,
)


class TestEdgeForceVirial(unittest.TestCase):
    def setUp(self) -> None:
        # 1 frame, 2 nodes, 2 real edges: e0 = (src=1, dst=0), e1 = (src=0, dst=1)
        self.edge_index = np.array([[1, 0], [0, 1]], dtype=np.int64)
        self.edge_vec = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.edge_mask = np.array([True, True])
        self.g = np.array([[0.5, 0.0, 0.0], [0.3, 0.0, 0.0]])  # per-edge grad
        self.n_node = np.array([2], dtype=np.int64)  # one frame, 2 nodes

    def test_force_formula(self) -> None:
        force, _, _ = edge_force_virial(
            self.g, self.edge_vec, self.edge_index, self.edge_mask, self.n_node
        )
        # F_k = sum_{dst=k} g - sum_{src=k} g
        # node 0: dst of e0 (+g0), src of e1 (-g1) => 0.5 - 0.3 = 0.2
        # node 1: dst of e1 (+g1), src of e0 (-g0) => 0.3 - 0.5 = -0.2
        np.testing.assert_allclose(force[:, 0], np.array([0.2, -0.2]))

    def test_atom_virial_full_to_src_sums_to_frame_virial(self) -> None:
        _, av, vir = edge_force_virial(
            self.g, self.edge_vec, self.edge_index, self.edge_mask, self.n_node
        )
        self.assertEqual(av.shape, (2, 3, 3))
        # all 2 nodes are in frame 0 => their atom-virials sum to that frame's virial
        np.testing.assert_allclose(np.sum(av, axis=0), vir[0])
        # full-to-src: e0 virial on node 1 (src), e1 virial on node 0 (src)
        w0 = -np.einsum("k,j->kj", self.g[0], self.edge_vec[0])
        w1 = -np.einsum("k,j->kj", self.g[1], self.edge_vec[1])
        np.testing.assert_allclose(av[1], w0)  # src of e0 is node 1
        np.testing.assert_allclose(av[0], w1)  # src of e1 is node 0

    def test_ragged_multiframe_with_edge_and_node_padding(self) -> None:
        # MOST GENERAL case: 2 frames with DIFFERENT node counts (3 and 5) AND
        # different edge counts (2 and 3), masked guard EDGES, and a padded NODE
        # axis (node_capacity 10 > sum(n_node)=8).
        n_node = np.array(
            [3, 5], dtype=np.int64
        )  # ragged: frame0={0,1,2}, frame1={3..7}
        node_capacity = 10  # 2 padded node slots (8, 9) at the global tail
        edge_index = np.array(
            [
                [1, 2, 4, 5, 6, 0, 0],  # src
                [0, 1, 3, 4, 7, 0, 0],
            ],  # dst  (frame0: dst 0,1 ; frame1: dst 3,4,7)
            dtype=np.int64,
        )
        edge_vec = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # frame 0 (2 edges)
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],  # frame 1 (3 edges)
                [9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0],  # masked guard edges
            ]
        )
        edge_mask = np.array([True, True, True, True, True, False, False])
        g = np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.7],
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [7.0, 7.0, 7.0],
                [7.0, 7.0, 7.0],
            ]
        )
        force, av, vir = edge_force_virial(
            g, edge_vec, edge_index, edge_mask, n_node, node_capacity=node_capacity
        )
        # shapes: padded node axis + per-frame virial
        self.assertEqual(force.shape, (10, 3))
        self.assertEqual(av.shape, (10, 3, 3))
        self.assertEqual(vir.shape, (2, 3, 3))
        # padded node slots (8, 9) are never referenced -> zero
        np.testing.assert_allclose(force[8:], 0.0)
        np.testing.assert_allclose(av[8:], 0.0)
        # per-frame virial = sum of THAT frame's real edges only (ragged edge counts)
        w = [-np.einsum("k,j->kj", g[i], edge_vec[i]) for i in range(5)]
        np.testing.assert_allclose(vir[0], w[0] + w[1])  # frame 0: edges 0,1
        np.testing.assert_allclose(vir[1], w[2] + w[3] + w[4])  # frame 1: edges 2,3,4
        self.assertFalse(np.allclose(vir[0], vir[1]))
        # per-frame atom-virial closure (ragged node blocks): frame nodes -> frame virial
        np.testing.assert_allclose(
            np.sum(av[0:3], axis=0), vir[0]
        )  # frame 0 nodes 0,1,2
        np.testing.assert_allclose(
            np.sum(av[3:8], axis=0), vir[1]
        )  # frame 1 nodes 3..7
        # guard edges contributed nothing: result == running with real edges only
        f2, a2, v2 = edge_force_virial(
            g[:5],
            edge_vec[:5],
            edge_index[:, :5],
            edge_mask[:5],
            n_node,
            node_capacity=node_capacity,
        )
        np.testing.assert_allclose(force, f2)
        np.testing.assert_allclose(av, a2)
        np.testing.assert_allclose(vir, v2)


if __name__ == "__main__":
    unittest.main()
