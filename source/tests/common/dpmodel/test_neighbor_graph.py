# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
)


class TestNeighborGraphDataclass(unittest.TestCase):
    def test_construct_minimal(self) -> None:
        ng = NeighborGraph(
            n_node=np.array([2], dtype=np.int64),
            edge_index=np.array([[1, 0], [0, 1]], dtype=np.int64),  # (2, E)
            edge_vec=np.zeros((2, 3), dtype=np.float64),
            edge_mask=np.array([True, True]),
        )
        self.assertEqual(ng.edge_index.shape, (2, 2))
        self.assertEqual(ng.edge_vec.shape, (2, 3))
        # optionals default to None
        self.assertIsNone(ng.n_local)
        self.assertIsNone(ng.angle_index)
        self.assertIsNone(ng.angle_mask)

    def test_graphlayout_defaults(self) -> None:
        lay = GraphLayout()
        self.assertIsNone(lay.edge_capacity)
        self.assertIsNone(lay.angle_capacity)
        self.assertIsNone(lay.node_capacity)
        self.assertIsNone(lay.frame_capacity)
        self.assertEqual(lay.min_edges, 2)


from deepmd.dpmodel.utils.neighbor_graph import (
    node_validity_mask,
)


class TestNodeValidityMask(unittest.TestCase):
    def test_no_padding_all_true(self) -> None:
        n_node = np.array([2, 3], dtype=np.int64)  # sum = 5
        mask = node_validity_mask(n_node, 5)
        np.testing.assert_array_equal(mask, np.array([True] * 5))

    def test_with_padding_prefix(self) -> None:
        n_node = np.array([2, 3], dtype=np.int64)  # 5 real
        mask = node_validity_mask(n_node, 8)  # N_max = 8 => 3 padding
        np.testing.assert_array_equal(mask, np.array([True] * 5 + [False] * 3))


from deepmd.dpmodel.utils.neighbor_graph import (
    pad_and_guard_edges,
)


class TestPadAndGuardEdges(unittest.TestCase):
    def setUp(self) -> None:
        self.edge_index = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.int64)  # E=3
        self.edge_vec = np.arange(9, dtype=np.float64).reshape(3, 3)

    def test_dynamic_appends_min_edges_guards(self) -> None:
        # capacity=None (torch): append min_edges masked dummies at the tail
        ei, ev, em = pad_and_guard_edges(
            self.edge_index, self.edge_vec, capacity=None, min_edges=2
        )
        self.assertEqual(ei.shape, (2, 5))  # 3 real + 2 guard
        self.assertEqual(ev.shape, (5, 3))
        np.testing.assert_array_equal(em, np.array([True, True, True, False, False]))
        # real edges unchanged at the front
        np.testing.assert_array_equal(ei[:, :3], self.edge_index)
        np.testing.assert_allclose(ev[:3], self.edge_vec)
        # guard edges are zero-vec, in-range index (pad_value=0)
        np.testing.assert_allclose(ev[3:], 0.0)
        np.testing.assert_array_equal(ei[:, 3:], 0)

    def test_static_capacity_pads_to_E_max(self) -> None:
        ei, ev, em = pad_and_guard_edges(
            self.edge_index, self.edge_vec, capacity=6, min_edges=2
        )
        self.assertEqual(ei.shape, (2, 6))
        np.testing.assert_array_equal(
            em, np.array([True, True, True, False, False, False])
        )

    def test_overflow_raises(self) -> None:
        with self.assertRaises(ValueError):
            pad_and_guard_edges(self.edge_index, self.edge_vec, capacity=2, min_edges=2)


class TestPublicExports(unittest.TestCase):
    def test_importable_from_utils(self) -> None:
        from deepmd.dpmodel.utils import (
            GraphLayout,
            NeighborGraph,
            NumpyNeighborList,
            edge_force_virial,
            segment_sum,
        )

        self.assertTrue(callable(segment_sum))
        self.assertTrue(callable(edge_force_virial))
        self.assertIsNotNone(NeighborGraph)
        self.assertIsNotNone(GraphLayout)
        self.assertIsNotNone(NumpyNeighborList)
