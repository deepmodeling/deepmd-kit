# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
    build_edge_csr,
    canonicalize_neighbor_graph,
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
        self.assertFalse(ng.destination_sorted)
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


class TestBuildEdgeCSR(unittest.TestCase):
    def test_preserves_payload_and_builds_stable_permutations(self) -> None:
        edge_vec = np.arange(15, dtype=np.float64).reshape(5, 3)
        edge_mask = np.array([True, False, True, True, False])
        for index_dtype in (np.int32, np.int64):
            with self.subTest(index_dtype=index_dtype):
                edge_index = np.array(
                    [[2, 0, 1, 2, 0], [1, 2, 0, 0, 1]],
                    dtype=index_dtype,
                )
                (
                    result_index,
                    result_vec,
                    result_mask,
                    destination_order,
                    destination_row_ptr,
                    source_row_ptr,
                    source_order,
                ) = build_edge_csr(edge_index, edge_vec, edge_mask, n_nodes=3)

                np.testing.assert_array_equal(result_index, edge_index)
                np.testing.assert_array_equal(result_vec, edge_vec)
                np.testing.assert_array_equal(result_mask, edge_mask)
                np.testing.assert_array_equal(
                    destination_order, np.array([2, 3, 0, 1, 4])
                )
                np.testing.assert_array_equal(
                    destination_row_ptr, np.array([0, 2, 3, 3])
                )
                np.testing.assert_array_equal(source_order, np.array([2, 0, 3, 1, 4]))
                np.testing.assert_array_equal(source_row_ptr, np.array([0, 0, 1, 3]))
                self.assertEqual(destination_order.dtype, index_dtype)
                self.assertEqual(source_order.dtype, index_dtype)
                self.assertEqual(destination_row_ptr.dtype, np.int64)
                self.assertEqual(source_row_ptr.dtype, np.int64)

    def test_canonicalizes_destination_payload_stably(self) -> None:
        edge_index = np.array([[2, 0, 1, 2, 0], [1, 2, 0, 0, 1]], dtype=np.int64)
        edge_vec = np.arange(15, dtype=np.float64).reshape(5, 3)
        edge_mask = np.array([True, False, True, True, False])
        (
            result_index,
            result_vec,
            result_mask,
            destination_order,
            destination_row_ptr,
            source_row_ptr,
            source_order,
        ) = build_edge_csr(
            edge_index,
            edge_vec,
            edge_mask,
            n_nodes=3,
            canonicalize=True,
        )

        expected_permutation = np.array([2, 3, 0, 1, 4])
        np.testing.assert_array_equal(result_index, edge_index[:, expected_permutation])
        np.testing.assert_array_equal(result_vec, edge_vec[expected_permutation])
        np.testing.assert_array_equal(
            result_mask, np.array([True, True, True, False, False])
        )
        np.testing.assert_array_equal(destination_order, np.arange(5))
        np.testing.assert_array_equal(destination_row_ptr, np.array([0, 2, 3, 3]))
        np.testing.assert_array_equal(source_order, np.arange(5))
        np.testing.assert_array_equal(source_row_ptr, np.array([0, 0, 1, 3]))

    def test_canonicalizes_neighbor_graph_at_deployment_boundary(self) -> None:
        edge_index = np.array([[2, 1, 0], [1, 0, 2]], dtype=np.int64)
        graph = NeighborGraph(
            n_node=np.array([3], dtype=np.int64),
            edge_index=edge_index,
            edge_vec=np.arange(9, dtype=np.float64).reshape(3, 3),
            edge_mask=np.array([True, True, True]),
        )

        result = canonicalize_neighbor_graph(graph, n_nodes=3)

        self.assertTrue(result.destination_sorted)
        np.testing.assert_array_equal(
            result.edge_index, edge_index[:, np.array([1, 0, 2])]
        )
        np.testing.assert_array_equal(result.destination_order, np.arange(3))

    def test_canonicalization_rebuilds_an_untrusted_canonical_claim(self) -> None:
        graph = NeighborGraph(
            n_node=np.array([2], dtype=np.int64),
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_vec=np.zeros((2, 3), dtype=np.float64),
            edge_mask=np.array([True, True]),
            destination_order=np.array([1, 0], dtype=np.int64),
            destination_row_ptr=np.array([0, 1, 2], dtype=np.int64),
            source_row_ptr=np.array([0, 1, 2], dtype=np.int64),
            source_order=np.array([0, 1], dtype=np.int64),
            destination_sorted=True,
        )

        result = canonicalize_neighbor_graph(graph, n_nodes=2)

        np.testing.assert_array_equal(result.destination_order, np.arange(2))
        np.testing.assert_array_equal(
            result.edge_index,
            np.array([[1, 0], [0, 1]], dtype=np.int64),
        )


from deepmd.dpmodel.utils.neighbor_graph import (
    node_ownership_mask,
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

    def test_local_plus_halo_ownership(self) -> None:
        n_node = np.array([3, 4], dtype=np.int64)
        n_local = np.array([2, 1], dtype=np.int64)
        mask = node_ownership_mask(n_node, n_local, 7)
        np.testing.assert_array_equal(
            mask,
            np.array([True, True, False, True, False, False, False]),
        )


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
            build_neighbor_graph,
            edge_force_virial,
            from_dense_quartet,
            segment_sum,
        )

        self.assertTrue(callable(segment_sum))
        self.assertTrue(callable(edge_force_virial))
        self.assertTrue(callable(build_neighbor_graph))
        self.assertTrue(callable(from_dense_quartet))
        self.assertIsNotNone(NeighborGraph)
        self.assertIsNotNone(GraphLayout)
