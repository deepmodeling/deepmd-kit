# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for static edge_capacity masked padding in build_neighbor_graph.

Codifies the contract: build_neighbor_graph(..., layout=GraphLayout(edge_capacity=E_max))
returns a NeighborGraph whose edge_index/edge_vec/edge_mask have a STATIC leading
edge dim E_max (real edges in the compact prefix, edge_mask=False tail), so export
sees a fixed E.  Edge overflow must raise ValueError.
"""

import numpy as np
import pytest

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    build_neighbor_graph,
)


class TestStaticEdgeCapacity:
    """Tests for static edge_capacity masked padding via build_neighbor_graph."""

    @pytest.fixture()
    def small_system(self):
        """6-atom periodic system with a 20 Å box (atoms well within rcut=4 range)."""
        rng = np.random.default_rng(0)
        coord = rng.normal(size=(1, 6, 3)) * 1.5
        atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
        box = np.eye(3).reshape(1, 9) * 20.0
        return coord, atype, box

    def test_static_edge_capacity_shape(self, small_system):
        """Static edge_capacity=64 yields edge_index.shape == (2, 64)."""
        coord, atype, box = small_system
        cap = build_neighbor_graph(
            coord, atype, box, 4.0, layout=GraphLayout(edge_capacity=64)
        )
        assert cap.edge_index.shape == (2, 64)
        assert cap.edge_vec.shape == (64, 3)
        assert cap.edge_mask.shape == (64,)

    def test_static_edge_capacity_matches_dynamic(self, small_system):
        """Static graph has same real-edge count as dynamic graph."""
        coord, atype, box = small_system
        dyn = build_neighbor_graph(coord, atype, box, 4.0)
        cap = build_neighbor_graph(
            coord, atype, box, 4.0, layout=GraphLayout(edge_capacity=64)
        )
        assert cap.edge_index.shape == (2, 64)
        assert int(cap.edge_mask.sum()) == int(dyn.edge_mask.sum())

    def test_static_edge_capacity_real_prefix_matches_dynamic(self, small_system):
        """The real-edge prefix of the static graph matches the dynamic graph."""
        coord, atype, box = small_system
        dyn = build_neighbor_graph(coord, atype, box, 4.0)
        cap = build_neighbor_graph(
            coord, atype, box, 4.0, layout=GraphLayout(edge_capacity=64)
        )
        n_real = int(dyn.edge_mask.sum())
        # real prefix must match exactly
        np.testing.assert_array_equal(
            cap.edge_index[:, :n_real], dyn.edge_index[:, :n_real]
        )
        np.testing.assert_allclose(cap.edge_vec[:n_real], dyn.edge_vec[:n_real])
        # padding suffix must have edge_mask=False
        assert not np.any(cap.edge_mask[n_real:])

    def test_overflow_raises(self, small_system):
        """edge_capacity smaller than real edge count must raise ValueError."""
        coord, atype, box = small_system
        # capacity=1 is guaranteed to be smaller than the real edge count
        with pytest.raises(ValueError, match="edge overflow"):
            build_neighbor_graph(
                coord, atype, box, 4.0, layout=GraphLayout(edge_capacity=1)
            )
