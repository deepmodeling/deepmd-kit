# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.edge_derivatives import (
    edge_force_virial,
)


class TestEdgeForceVirial(unittest.TestCase):
    def setUp(self) -> None:
        # 2 nodes, 2 real edges: e0 = (src=1, dst=0), e1 = (src=0, dst=1)
        self.edge_index = np.array([[1, 0], [0, 1]], dtype=np.int64)
        self.edge_vec = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.edge_mask = np.array([True, True])
        self.g = np.array([[0.5, 0.0, 0.0], [0.3, 0.0, 0.0]])  # per-edge grad
        self.N = 2

    def test_force_formula(self) -> None:
        force, _, _ = edge_force_virial(
            self.g, self.edge_vec, self.edge_index, self.edge_mask, self.N
        )
        # F_k = sum_{dst=k} g - sum_{src=k} g
        # node 0: dst of e0 (+g0), src of e1 (-g1) => 0.5 - 0.3 = 0.2
        # node 1: dst of e1 (+g1), src of e0 (-g0) => 0.3 - 0.5 = -0.2
        np.testing.assert_allclose(force[:, 0], np.array([0.2, -0.2]))

    def test_global_virial_is_sum_of_edge_outer(self) -> None:
        _, _, gv = edge_force_virial(
            self.g, self.edge_vec, self.edge_index, self.edge_mask, self.N
        )
        # W = -sum_e g_e (x) edge_vec_e
        want = -(
            np.einsum("k,j->kj", self.g[0], self.edge_vec[0])
            + np.einsum("k,j->kj", self.g[1], self.edge_vec[1])
        )
        np.testing.assert_allclose(gv, want)

    def test_atom_virial_full_to_src_sums_to_global(self) -> None:
        _, av, gv = edge_force_virial(
            self.g, self.edge_vec, self.edge_index, self.edge_mask, self.N
        )
        self.assertEqual(av.shape, (2, 3, 3))
        np.testing.assert_allclose(np.sum(av, axis=0), gv)
        # full-to-src: e0 virial on node 1 (src), e1 virial on node 0 (src)
        w0 = -np.einsum("k,j->kj", self.g[0], self.edge_vec[0])
        w1 = -np.einsum("k,j->kj", self.g[1], self.edge_vec[1])
        np.testing.assert_allclose(av[1], w0)  # src of e0 is node 1
        np.testing.assert_allclose(av[0], w1)  # src of e1 is node 0

    def test_padding_edges_contribute_nothing(self) -> None:
        # append a masked guard edge pointing at node 0 with nonzero g (should be ignored)
        ei = np.concatenate([self.edge_index, np.array([[0], [0]], dtype=np.int64)], axis=1)
        ev = np.concatenate([self.edge_vec, np.array([[9.0, 9.0, 9.0]])], axis=0)
        em = np.array([True, True, False])
        g = np.concatenate([self.g, np.array([[7.0, 7.0, 7.0]])], axis=0)
        f1, a1, v1 = edge_force_virial(g, ev, ei, em, self.N)
        f0, a0, v0 = edge_force_virial(self.g, self.edge_vec, self.edge_index, self.edge_mask, self.N)
        np.testing.assert_allclose(f1, f0)
        np.testing.assert_allclose(a1, a0)
        np.testing.assert_allclose(v1, v0)
