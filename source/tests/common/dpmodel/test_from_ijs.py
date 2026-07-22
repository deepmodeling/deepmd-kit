# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import pytest

from deepmd.dpmodel.utils.neighbor_graph import (
    neighbor_graph_from_ijs,
)


class TestFromIjs(unittest.TestCase):
    def test_edge_vec_and_index(self) -> None:
        """src=j, dst=i, edge_vec = coord[j] + S@box - coord[i] (single frame, S=0)."""
        coord = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]]])  # (1,3,3)
        box = np.eye(3)[None] * 6.0
        i = np.array([0, 1])  # center
        j = np.array([1, 0])  # neighbor
        S = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64)
        ng = neighbor_graph_from_ijs(
            i, j, S, coord, box, nframe_id=np.zeros(2, np.int64), nloc=3
        )
        np.testing.assert_array_equal(ng.edge_index[0][ng.edge_mask], j)  # src
        np.testing.assert_array_equal(ng.edge_index[1][ng.edge_mask], i)  # dst
        np.testing.assert_allclose(
            ng.edge_vec[ng.edge_mask][0], coord[0, 1] - coord[0, 0]
        )

    def test_periodic_shift_in_edge_vec(self) -> None:
        """A nonzero S contributes S@box to edge_vec (image neighbor)."""
        coord = np.array([[[0.5, 0, 0], [5.5, 0, 0]]])  # (1,2,3)
        box = np.eye(3)[None] * 6.0
        i = np.array([0])
        j = np.array([1])
        S = np.array([[-1, 0, 0]], dtype=np.int64)
        ng = neighbor_graph_from_ijs(
            i, j, S, coord, box, nframe_id=np.zeros(1, np.int64), nloc=2
        )
        # coord[1] + (-1,0,0)@box - coord[0] = 5.5 - 6 - 0.5 = -1.0
        np.testing.assert_allclose(
            ng.edge_vec[ng.edge_mask][0], np.array([-1.0, 0.0, 0.0])
        )


class TestAseCarryAll(unittest.TestCase):
    def _sets(self, ng, nloc):
        # per-center set of (src, rounded edge_vec); real edges only
        ei = ng.edge_index[:, ng.edge_mask]
        ev = ng.edge_vec[ng.edge_mask]
        s = [set() for _ in range(nloc)]
        for k in range(ei.shape[1]):
            s[int(ei[1, k])].add((int(ei[0, k]), tuple(np.round(ev[k], 6))))
        return s

    def test_ase_matches_intree_carry_all(self) -> None:
        """ASE carry-all builder yields the SAME neighbor set as the in-tree
        carry-all build_neighbor_graph (both carry ALL neighbors in rcut).
        """
        pytest.importorskip("ase")
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
            build_neighbor_graph_ase,
        )

        rng = np.random.default_rng(3)
        coord = rng.normal(size=(1, 8, 3)) * 2.0
        atype = np.array([[0, 1] * 4], dtype=np.int64)
        box = np.eye(3)[None] * 8.0
        ng_ase = build_neighbor_graph_ase(coord, atype, box, rcut=4.0)
        ng_ref = build_neighbor_graph(coord, atype, box, rcut=4.0)
        self.assertEqual(self._sets(ng_ase, 8), self._sets(ng_ref, 8))

    def test_ase_matches_intree_carry_all_nonperiodic(self) -> None:
        """Non-periodic (box=None): ASE carry-all == in-tree carry-all."""
        pytest.importorskip("ase")
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
            build_neighbor_graph_ase,
        )

        rng = np.random.default_rng(7)
        coord = rng.normal(size=(1, 6, 3)) * 2.0
        atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
        ng_ase = build_neighbor_graph_ase(coord, atype, None, rcut=4.0)
        ng_ref = build_neighbor_graph(coord, atype, None, rcut=4.0)
        self.assertEqual(self._sets(ng_ase, 6), self._sets(ng_ref, 6))

    def test_ase_excludes_virtual_atoms_like_dense(self) -> None:
        """Virtual atoms (atype < 0) excluded as center AND neighbor.

        The dense reference builder filters virtual atoms during construction;
        the geometric ASE search is type-blind, so the builder must post-filter
        to keep the World-2 "same neighbor set" contract (OutisLi, #5714).
        """
        pytest.importorskip("ase")
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
            build_neighbor_graph_ase,
        )

        rng = np.random.default_rng(5)
        coord = rng.normal(size=(1, 8, 3)) * 2.0
        atype = np.array([[0, 1, -1, 0, 1, -1, 0, 1]], dtype=np.int64)
        box = np.eye(3)[None] * 8.0
        ng_ase = build_neighbor_graph_ase(coord, atype, box, rcut=4.0)
        ng_ref = build_neighbor_graph(coord, atype, box, rcut=4.0)
        self.assertEqual(self._sets(ng_ase, 8), self._sets(ng_ref, 8))
        # no real edge touches a virtual atom
        ei = ng_ase.edge_index[:, ng_ase.edge_mask]
        flat_atype = atype.reshape(-1)
        assert np.all(flat_atype[ei[0]] >= 0) and np.all(flat_atype[ei[1]] >= 0)


if __name__ == "__main__":
    unittest.main()
