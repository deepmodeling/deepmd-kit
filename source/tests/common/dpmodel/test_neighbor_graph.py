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
        np.testing.assert_array_equal(
            mask, np.array([True] * 5 + [False] * 3)
        )
