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
