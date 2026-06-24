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

    def test_virial_is_per_frame(self) -> None:
        _, _, vir = edge_force_virial(
            self.g, self.edge_vec, self.edge_index, self.edge_mask, self.n_node
        )
        # single frame => shape (1, 3, 3); value = -sum_e g_e (x) edge_vec_e
        self.assertEqual(vir.shape, (1, 3, 3))
        want = -(
            np.einsum("k,j->kj", self.g[0], self.edge_vec[0])
            + np.einsum("k,j->kj", self.g[1], self.edge_vec[1])
        )
        np.testing.assert_allclose(vir[0], want)

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

    def test_padding_edges_contribute_nothing(self) -> None:
        # append a masked guard edge pointing at node 0 with nonzero g (ignored)
        ei = np.concatenate(
            [self.edge_index, np.array([[0], [0]], dtype=np.int64)], axis=1
        )
        ev = np.concatenate([self.edge_vec, np.array([[9.0, 9.0, 9.0]])], axis=0)
        em = np.array([True, True, False])
        g = np.concatenate([self.g, np.array([[7.0, 7.0, 7.0]])], axis=0)
        f1, a1, v1 = edge_force_virial(g, ev, ei, em, self.n_node)
        f0, a0, v0 = edge_force_virial(
            self.g, self.edge_vec, self.edge_index, self.edge_mask, self.n_node
        )
        np.testing.assert_allclose(f1, f0)
        np.testing.assert_allclose(a1, a0)
        np.testing.assert_allclose(v1, v0)

    def test_multiframe_virials_not_collapsed(self) -> None:
        # 2 frames, 2 nodes each: nodes {0,1} in frame 0, {2,3} in frame 1.
        # Frame edges chosen so the two frames have DISTINCT virials; the bug
        # (summing all edges into one (3,3)) would merge them.
        n_node = np.array([2, 2], dtype=np.int64)  # N = 4
        # frame 0 edges: e0=(src=1,dst=0), e1=(src=0,dst=1); frame 1: e2=(3,2), e3=(2,3)
        edge_index = np.array([[1, 0, 3, 2], [0, 1, 2, 3]], dtype=np.int64)
        edge_vec = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],  # frame 1 uses a different direction/scale
                [0.0, -2.0, 0.0],
            ]
        )
        edge_mask = np.array([True, True, True, True])
        g = np.array(
            [
                [0.5, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.0, 0.7, 0.0],
                [0.0, 0.1, 0.0],
            ]
        )
        force, av, vir = edge_force_virial(g, edge_vec, edge_index, edge_mask, n_node)
        self.assertEqual(vir.shape, (2, 3, 3))  # per-frame, NOT collapsed
        w = [-np.einsum("k,j->kj", g[i], edge_vec[i]) for i in range(4)]
        np.testing.assert_allclose(vir[0], w[0] + w[1])  # frame 0 edges only
        np.testing.assert_allclose(vir[1], w[2] + w[3])  # frame 1 edges only
        # the two frames are genuinely different (would be equal-ish if merged-then-split wrong)
        self.assertFalse(np.allclose(vir[0], vir[1]))
        # per-frame atom-virial closure: frame-f nodes' atom-virials sum to vir[f]
        np.testing.assert_allclose(np.sum(av[0:2], axis=0), vir[0])
        np.testing.assert_allclose(np.sum(av[2:4], axis=0), vir[1])
        # force is per-node (flat across frames), unaffected
        self.assertEqual(force.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
