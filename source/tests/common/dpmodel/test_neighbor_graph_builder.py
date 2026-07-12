# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the dpmodel NeighborGraph builder/converter.

``build_neighbor_graph`` is the CARRY-ALL ``dense`` search backend: it builds a
graph DIRECTLY from coordinates and keeps EVERY neighbor within ``rcut`` (no
``sel`` truncation). We validate it against an INDEPENDENT brute-force all-pairs
oracle defined locally in this test file.

``from_dense_quartet`` is the backward-compat CONVERTER: it adapts an existing
(``sel``-truncated) extended quartet and performs no search.
"""

import itertools
import unittest

import numpy as np

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    apply_pair_exclusion,
    build_neighbor_graph,
    from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


def brute_force_neighbor_sets(coord, box, rcut):
    """Independent all-pairs oracle: per center i, the set of (local-owner j,
    rounded edge_vec) within rcut. edge_vec = coord[j] + S@box - coord[i].
    """
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


def graph_neighbor_sets(ng, nloc):
    """Per dst-center, the set of (src local owner, rounded edge_vec); real edges only."""
    ei = ng.edge_index[:, ng.edge_mask]
    ev = ng.edge_vec[ng.edge_mask]
    sets = [set() for _ in range(nloc)]
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        sets[dst].add((src, tuple(np.round(ev[k], 6))))
    return sets


def graph_neighbor_sets_frame(ng, frame, nloc):
    """Per-frame neighbor sets (src/dst de-offset to local [0, nloc)); real edges only.

    Selects the edges whose dst lives in frame ``frame``'s node block
    ``[frame*nloc, (frame+1)*nloc)`` and de-offsets indices, so the result is
    directly comparable to a single-frame oracle.
    """
    off = frame * nloc
    ei = ng.edge_index[:, ng.edge_mask]
    ev = ng.edge_vec[ng.edge_mask]
    sets = [set() for _ in range(nloc)]
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        if off <= dst < off + nloc:
            sets[dst - off].add((src - off, tuple(np.round(ev[k], 6))))
    return sets


class TestNeighborGraphBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.rcut = 4.0
        # atom 2 at y=2.3 (not 2.0): avoids a degenerate pair sitting exactly at
        # rcut under PBC (box 6, image distance 6-2=4==rcut), where strict-< vs
        # <= cutoff conventions disagree. Real geometries never sit exactly at rcut.
        self.coord = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.3, 0.0], [3.5, 0.0, 0.0]],
            dtype=np.float64,
        ).reshape(1, 4, 3)
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    def test_nonperiodic_matches_brute_force(self) -> None:
        ng = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        np.testing.assert_array_equal(ng.n_node, np.array([4], dtype=np.int64))
        self.assertEqual(
            graph_neighbor_sets(ng, 4),
            brute_force_neighbor_sets(self.coord[0], None, self.rcut),
        )

    def test_canonicalization_is_explicit(self) -> None:
        generic = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        with_csr = build_neighbor_graph(
            self.coord,
            self.atype,
            None,
            self.rcut,
            with_csr=True,
        )
        canonical = build_neighbor_graph(
            self.coord,
            self.atype,
            None,
            self.rcut,
            canonicalize=True,
        )

        self.assertFalse(generic.destination_sorted)
        self.assertIsNone(generic.destination_order)
        self.assertIsNone(generic.destination_row_ptr)
        self.assertIsNone(generic.source_row_ptr)
        self.assertIsNone(generic.source_order)
        self.assertFalse(with_csr.destination_sorted)
        self.assertIsNotNone(with_csr.destination_order)
        self.assertIsNotNone(with_csr.destination_row_ptr)
        self.assertIsNotNone(with_csr.source_row_ptr)
        self.assertIsNotNone(with_csr.source_order)
        self.assertTrue(canonical.destination_sorted)
        np.testing.assert_array_equal(
            canonical.destination_order,
            np.arange(canonical.edge_index.shape[1]),
        )
        real_destination = canonical.edge_index[1, canonical.edge_mask]
        self.assertTrue(bool(np.all(real_destination[:-1] <= real_destination[1:])))
        self.assertEqual(
            graph_neighbor_sets(canonical, 4),
            graph_neighbor_sets(generic, 4),
        )

    def test_periodic_matches_brute_force(self) -> None:
        box = np.eye(3, dtype=np.float64)[None] * 6.0
        ng = build_neighbor_graph(self.coord, self.atype, box, self.rcut)
        self.assertEqual(
            graph_neighbor_sets(ng, 4),
            brute_force_neighbor_sets(self.coord[0], box[0], self.rcut),
        )

    def test_neighbor_only_across_periodic_boundary(self) -> None:
        # DISCRIMINATING PBC case: a pair that is a neighbor ONLY across the
        # boundary. atoms at x=0.5 and x=5.5 in a box of 6: direct distance 5.0 >
        # rcut=4 (NOT a direct neighbor), but the minimum image is 1.0 < rcut.
        # A build that ignored periodic images would find ZERO edges here.
        box = np.eye(3, dtype=np.float64)[None] * 6.0
        coord = np.array([[0.5, 0.0, 0.0], [5.5, 0.0, 0.0]], dtype=np.float64).reshape(
            1, 2, 3
        )
        atype = np.array([[0, 0]], dtype=np.int64)
        ng = build_neighbor_graph(coord, atype, box, self.rcut)
        got = graph_neighbor_sets(ng, 2)  # per-center list of neighbor sets
        # each atom's ONLY neighbor is the other's periodic image, at +-1.0
        want = [{(1, (-1.0, 0.0, 0.0))}, {(0, (1.0, 0.0, 0.0))}]
        self.assertEqual(got, want)
        # the direct (non-image) separation of 5.0 must NOT appear as an edge
        ev = ng.edge_vec[ng.edge_mask]
        self.assertFalse(bool(np.any(np.linalg.norm(ev, axis=1) > 4.0)))
        # independent brute-force oracle agrees on the cross-boundary environment
        self.assertEqual(got, brute_force_neighbor_sets(coord[0], box[0], self.rcut))
        # and WITHOUT the box the same atoms are NOT neighbors (direct 5.0 > rcut)
        ng_free = build_neighbor_graph(coord, atype, None, self.rcut)
        self.assertEqual(int(ng_free.edge_mask.sum()), 0)

    def test_edge_vec_within_rcut(self) -> None:
        ng = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        ev = ng.edge_vec[ng.edge_mask]
        self.assertTrue(np.all(np.linalg.norm(ev, axis=1) < self.rcut))

    def test_carry_all_keeps_more_than_truncated_quartet(self) -> None:
        # THE carry-all contract: with a binding ``sel``, the legacy quartet
        # converter drops real neighbors, but the dense search keeps them all.
        box = np.eye(3, dtype=np.float64)[None] * 6.0
        # sel=1 per type -> heavily truncates under PBC (many images within rcut).
        ext_coord, _ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            self.coord, self.atype, self.rcut, [1, 1], mixed_types=True, box=box
        )
        ng_trunc = from_dense_quartet(ext_coord, nlist, mapping)
        ng_all = build_neighbor_graph(self.coord, self.atype, box, self.rcut)
        n_trunc = int(ng_trunc.edge_mask.sum())
        n_all = int(ng_all.edge_mask.sum())
        n_oracle = sum(
            len(s) for s in brute_force_neighbor_sets(self.coord[0], box[0], self.rcut)
        )
        # the truncated converter loses edges; the carry-all search recovers them all
        self.assertLess(n_trunc, n_all)
        self.assertEqual(n_all, n_oracle)

    def test_multiframe_per_frame_neighbor_sets(self) -> None:
        # TWO DIFFERENT frames -> different per-frame EDGE counts. (Node counts are
        # equal because build_neighbor_graph takes a rectangular (nf,nloc,3) coord;
        # ragged node counts need a future ragged builder and are exercised on the
        # flat primitives, e.g. test_edge_force_virial multi-frame.)
        coord_b = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.5, 1.5, 0.0]],
            dtype=np.float64,
        ).reshape(1, 4, 3)
        coord2 = np.concatenate([self.coord, coord_b], axis=0)  # (2,4,3), DIFFERENT
        atype2 = np.concatenate([self.atype, self.atype], axis=0)
        ng = build_neighbor_graph(coord2, atype2, None, self.rcut)
        np.testing.assert_array_equal(ng.n_node, np.array([4, 4], dtype=np.int64))
        # each frame's edges match THAT frame's own brute-force oracle
        self.assertEqual(
            graph_neighbor_sets_frame(ng, 0, 4),
            brute_force_neighbor_sets(coord2[0], None, self.rcut),
        )
        self.assertEqual(
            graph_neighbor_sets_frame(ng, 1, 4),
            brute_force_neighbor_sets(coord2[1], None, self.rcut),
        )
        # the two frames are genuinely different environments (different edge sets)
        self.assertNotEqual(
            graph_neighbor_sets_frame(ng, 0, 4),
            graph_neighbor_sets_frame(ng, 1, 4),
        )
        # node-offset invariant: frame-0 edges in [0,4), frame-1 in [4,8)
        ei = ng.edge_index[:, ng.edge_mask]
        self.assertTrue(np.all(ei[:, ei[1] < 4] < 4))
        self.assertTrue(np.all(ei[:, ei[1] >= 4] >= 4))

    def test_multiframe_periodic_per_frame(self) -> None:
        box = np.eye(3, dtype=np.float64)[None] * 6.0
        coord2 = np.concatenate([self.coord, self.coord + 0.3], axis=0)  # different
        atype2 = np.concatenate([self.atype, self.atype], axis=0)
        box2 = np.concatenate([box, box], axis=0)
        ng = build_neighbor_graph(coord2, atype2, box2, self.rcut)
        for f in (0, 1):
            self.assertEqual(
                graph_neighbor_sets_frame(ng, f, 4),
                brute_force_neighbor_sets(coord2[f], box2[f], self.rcut),
            )

    def test_virtual_atoms_excluded(self) -> None:
        # a virtual atom (type < 0) is excluded BOTH as a center (dst) and as a
        # neighbor (src). atom 0 (origin) has in-range neighbors 1 (dist 1.0) and
        # 2 (dist 2.3), so making it virtual actively exercises center-exclusion:
        # without the virtual-center guard, edges 0<-1 and 0<-2 would appear.
        atype = np.array([[-1, 1, 0, 1]], dtype=np.int64)  # atom 0 virtual
        ng = build_neighbor_graph(self.coord, atype, None, self.rcut)
        ei = ng.edge_index[:, ng.edge_mask]
        src, dst = ei[0], ei[1]
        self.assertFalse(bool(np.any(dst == 0)))  # never a center (center exclusion)
        self.assertFalse(bool(np.any(src == 0)))  # never a neighbor (neighbor excl.)
        # the remaining real atoms still neighbor each other (we didn't nuke all edges)
        self.assertGreater(int(ng.edge_mask.sum()), 0)

    def test_min_edges_guard_pads_sparse_frame(self) -> None:
        # a single isolated atom yields ZERO real edges; the dynamic (capacity=None)
        # layout must still emit the min_edges=2 guard edges, all masked out.
        coord = np.zeros((1, 1, 3), dtype=np.float64)
        atype = np.array([[0]], dtype=np.int64)
        ng = build_neighbor_graph(coord, atype, None, self.rcut)  # default layout
        self.assertEqual(ng.edge_index.shape[1], 2)  # min_edges guard edges
        self.assertEqual(int(ng.edge_mask.sum()), 0)  # none real
        self.assertTrue(np.all(ng.edge_vec == 0.0))

    def test_flat_coord_input_matches_rectangular(self) -> None:
        # coord given flattened (nf, nloc*3) must match the (nf, nloc, 3) form.
        coord_flat = self.coord.reshape(1, 4 * 3)
        ng_flat = build_neighbor_graph(coord_flat, self.atype, None, self.rcut)
        ng_rect = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        self.assertEqual(
            graph_neighbor_sets(ng_flat, 4), graph_neighbor_sets(ng_rect, 4)
        )

    def test_static_capacity_padding(self) -> None:
        ng = build_neighbor_graph(
            self.coord,
            self.atype,
            None,
            self.rcut,
            layout=GraphLayout(edge_capacity=64),
        )
        self.assertEqual(ng.edge_index.shape[1], 64)
        self.assertEqual(ng.edge_vec.shape[0], 64)
        # exactly the real edges are marked, padded compactly at the tail
        n_real = sum(
            len(s) for s in brute_force_neighbor_sets(self.coord[0], None, self.rcut)
        )
        self.assertEqual(int(ng.edge_mask.sum()), n_real)
        self.assertTrue(bool(np.all(ng.edge_mask[:n_real])))
        self.assertFalse(bool(np.any(ng.edge_mask[n_real:])))
        # masked-out tail contributes no real edges
        self.assertTrue(np.all(ng.edge_vec[~ng.edge_mask] == 0.0))


class TestFromDenseQuartet(unittest.TestCase):
    def test_adapter_on_handmade_quartet(self) -> None:
        # 2 local atoms, no ghosts; each is the other's only neighbor.
        extended_coord = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # (1,2,3)
        nlist = np.array([[[1, -1], [0, -1]]], dtype=np.int64)  # (1,2,2)
        mapping = np.array([[0, 1]], dtype=np.int64)  # (1,2) local->self
        ng = from_dense_quartet(extended_coord, nlist, mapping)
        ei = ng.edge_index[:, ng.edge_mask]
        ev = ng.edge_vec[ng.edge_mask]
        got = {
            (int(ei[0, k]), int(ei[1, k]), tuple(np.round(ev[k], 6)))
            for k in range(ei.shape[1])
        }
        want = {
            (1, 0, (1.0, 0.0, 0.0)),  # center 0, neighbor 1, vec = r1 - r0
            (0, 1, (-1.0, 0.0, 0.0)),  # center 1, neighbor 0, vec = r0 - r1
        }
        self.assertEqual(got, want)

    def test_adapter_multiframe_offsets(self) -> None:
        # 2 frames, 2 local atoms each; each atom's only neighbor is the other.
        # Frame 1 has a different separation so its edge_vec differs.
        extended_coord = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # frame 0
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # frame 1 (different)
            ]
        )
        nlist = np.array(
            [[[1, -1], [0, -1]], [[1, -1], [0, -1]]], dtype=np.int64
        )  # (2,2,2)
        mapping = np.array([[0, 1], [0, 1]], dtype=np.int64)
        ng = from_dense_quartet(extended_coord, nlist, mapping)
        np.testing.assert_array_equal(ng.n_node, np.array([2, 2], dtype=np.int64))
        ei = ng.edge_index[:, ng.edge_mask]
        ev = ng.edge_vec[ng.edge_mask]
        per = {}
        for k in range(ei.shape[1]):
            per[(int(ei[0, k]), int(ei[1, k]))] = tuple(np.round(ev[k], 6))
        # frame 0 nodes {0,1} with sep 1.0; frame 1 nodes {2,3} with sep 2.0
        self.assertEqual(per[(1, 0)], (1.0, 0.0, 0.0))
        self.assertEqual(per[(0, 1)], (-1.0, 0.0, 0.0))
        self.assertEqual(per[(3, 2)], (2.0, 0.0, 0.0))
        self.assertEqual(per[(2, 3)], (-2.0, 0.0, 0.0))

    def test_adapter_maps_ghost_to_local_owner(self) -> None:
        # 1 local atom (0) + 1 ghost (1) which is a periodic image of atom 0.
        extended_coord = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])  # (1,2,3)
        nlist = np.array([[[1, -1]]], dtype=np.int64)  # (1, nloc=1, nsel=2)
        mapping = np.array([[0, 0]], dtype=np.int64)  # ghost 1 -> owner 0
        ng = from_dense_quartet(extended_coord, nlist, mapping)
        ei = ng.edge_index[:, ng.edge_mask]
        ev = ng.edge_vec[ng.edge_mask]
        self.assertEqual(ei.shape[1], 1)
        # src = local owner of the ghost (0), dst = center (0); vec carries the shift
        self.assertEqual((int(ei[0, 0]), int(ei[1, 0])), (0, 0))
        np.testing.assert_allclose(ev[0], np.array([3.0, 0.0, 0.0]))


def valid_edge_set(ng):
    """Return the set of (src, dst, rounded edge_vec) for all real edges."""
    ei = ng.edge_index[:, ng.edge_mask]
    ev = ng.edge_vec[ng.edge_mask]
    return {
        (int(ei[0, k]), int(ei[1, k]), tuple(np.round(ev[k], 6)))
        for k in range(ei.shape[1])
    }


class TestBuildNeighborGraphPairExclOracle(unittest.TestCase):
    """Oracle harness: builder(pair_excl=X) == builder() + apply_pair_exclusion(X).

    Covers both ``pair_excl=None`` (identity; no exclusion applied) and a
    non-empty exclusion set, for the ``dense`` backend.  The oracle asserts
    SET-EQUALITY of the valid-edge set, matching the Task 3b contract.
    """

    def setUp(self) -> None:
        self.rcut = 4.0
        # 4 atoms, 2 types (0 and 1); atom 2 offset avoids degenerate rcut alignment.
        self.coord = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.3, 0.0], [3.5, 0.0, 0.0]],
            dtype=np.float64,
        ).reshape(1, 4, 3)
        # type sequence: 0,1,0,1 -- pairs (0,1) and (1,0) are heterogeneous
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)
        self.ntypes = 2

    def _pair_excl(self, exclude_pairs):
        """Build a PairExcludeMask from a list of (ti, tj) tuples."""
        return PairExcludeMask(self.ntypes, exclude_pairs)

    def test_pair_excl_none_identity_dense(self) -> None:
        """pair_excl=None: builder output unchanged (identity)."""
        ng_ref = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        ng_excl = build_neighbor_graph(
            self.coord, self.atype, None, self.rcut, pair_excl=None
        )
        self.assertEqual(valid_edge_set(ng_ref), valid_edge_set(ng_excl))

    def test_pair_excl_empty_list_identity_dense(self) -> None:
        """pair_excl with empty exclude set: builder output unchanged."""
        pe = self._pair_excl([])
        ng_ref = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        ng_excl = build_neighbor_graph(
            self.coord, self.atype, None, self.rcut, pair_excl=pe
        )
        self.assertEqual(valid_edge_set(ng_ref), valid_edge_set(ng_excl))

    def test_oracle_set_equality_dense_nonperiodic(self) -> None:
        """Builder with pair_excl==(0,1) == builder() + apply_pair_exclusion."""
        pe = self._pair_excl([(0, 1), (1, 0)])
        # reference: build without exclusion then apply separately
        ng_base = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        atype_flat = self.atype.reshape(-1)
        ng_post = apply_pair_exclusion(ng_base, atype_flat, pe)
        # under test: builder applies exclusion internally
        ng_fused = build_neighbor_graph(
            self.coord, self.atype, None, self.rcut, pair_excl=pe
        )
        self.assertEqual(valid_edge_set(ng_post), valid_edge_set(ng_fused))
        # sanity: exclusion actually REMOVED some edges
        self.assertLess(int(ng_fused.edge_mask.sum()), int(ng_base.edge_mask.sum()))

    def test_canonical_csr_reflects_pair_exclusion(self) -> None:
        pe = self._pair_excl([(0, 1), (1, 0)])
        graph = build_neighbor_graph(
            self.coord,
            self.atype,
            None,
            self.rcut,
            pair_excl=pe,
            canonicalize=True,
        )

        valid_edges = int(graph.edge_mask.sum())
        self.assertEqual(int(graph.destination_row_ptr[-1]), valid_edges)
        self.assertEqual(int(graph.source_row_ptr[-1]), valid_edges)
        self.assertTrue(bool(np.all(graph.edge_mask[:valid_edges])))
        self.assertFalse(bool(np.any(graph.edge_mask[valid_edges:])))
        destination = graph.edge_index[1, :valid_edges]
        self.assertTrue(bool(np.all(destination[:-1] <= destination[1:])))

    def test_oracle_set_equality_dense_periodic(self) -> None:
        """Periodic PBC: builder with pair_excl==(0,0) == builder() + apply."""
        pe = self._pair_excl([(0, 0)])
        box = np.eye(3, dtype=np.float64)[None] * 6.0
        ng_base = build_neighbor_graph(self.coord, self.atype, box, self.rcut)
        atype_flat = self.atype.reshape(-1)
        ng_post = apply_pair_exclusion(ng_base, atype_flat, pe)
        ng_fused = build_neighbor_graph(
            self.coord, self.atype, box, self.rcut, pair_excl=pe
        )
        self.assertEqual(valid_edge_set(ng_post), valid_edge_set(ng_fused))
        # type-0 centers: atoms 0,2; type-0 neighbors excluded; fewer edges expected
        self.assertLess(int(ng_fused.edge_mask.sum()), int(ng_base.edge_mask.sum()))

    def test_oracle_set_equality_dense_multiframe(self) -> None:
        """Multi-frame: set-equality holds per frame."""
        pe = self._pair_excl([(0, 1), (1, 0)])
        coord2 = np.concatenate([self.coord, self.coord + 0.5], axis=0)
        atype2 = np.concatenate([self.atype, self.atype], axis=0)
        ng_base = build_neighbor_graph(coord2, atype2, None, self.rcut)
        atype_flat = atype2.reshape(-1)
        ng_post = apply_pair_exclusion(ng_base, atype_flat, pe)
        ng_fused = build_neighbor_graph(coord2, atype2, None, self.rcut, pair_excl=pe)
        self.assertEqual(valid_edge_set(ng_post), valid_edge_set(ng_fused))


class TestBuildNeighborGraphAseOracle(unittest.TestCase):
    """Oracle harness for the ASE builder pair_excl parameter.

    Skipped when ``ase`` is not installed.  Asserts set-equality of the
    valid-edge set between the ASE builder called with ``pair_excl`` and
    the dense reference builder + separate :func:`apply_pair_exclusion`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import ase  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest("ase not installed") from e

    def setUp(self) -> None:
        self.rcut = 4.0
        self.coord = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.3, 0.0], [3.5, 0.0, 0.0]],
            dtype=np.float64,
        ).reshape(1, 4, 3)
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)
        self.ntypes = 2

    def _pair_excl(self, exclude_pairs):
        return PairExcludeMask(self.ntypes, exclude_pairs)

    def test_ase_pair_excl_none_identity(self) -> None:
        """pair_excl=None: ASE builder output unchanged."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph_ase,
        )

        ng_ref = build_neighbor_graph_ase(self.coord, self.atype, None, self.rcut)
        ng_excl = build_neighbor_graph_ase(
            self.coord, self.atype, None, self.rcut, pair_excl=None
        )
        self.assertEqual(valid_edge_set(ng_ref), valid_edge_set(ng_excl))

    def test_ase_oracle_set_equality(self) -> None:
        """ASE builder with pair_excl == dense ref + apply_pair_exclusion."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph_ase,
        )

        pe = self._pair_excl([(0, 1), (1, 0)])
        # dense reference + separate post-process
        ng_dense = build_neighbor_graph(self.coord, self.atype, None, self.rcut)
        atype_flat = self.atype.reshape(-1)
        ng_ref = apply_pair_exclusion(ng_dense, atype_flat, pe)
        # ASE builder with fused post-process
        ng_ase = build_neighbor_graph_ase(
            self.coord, self.atype, None, self.rcut, pair_excl=pe
        )
        self.assertEqual(valid_edge_set(ng_ref), valid_edge_set(ng_ase))
        # exclusion actually removed edges
        ng_ase_plain = build_neighbor_graph_ase(self.coord, self.atype, None, self.rcut)
        self.assertLess(int(ng_ase.edge_mask.sum()), int(ng_ase_plain.edge_mask.sum()))


# NOTE: nvalchemiops builder has no local oracle set-equality test for pair_excl
# because it requires CUDA; validation is deferred to GPU box tests (PR-C/nv-gtest).
# See deepmd.pt_expt.utils.nv_graph_builder.build_neighbor_graph_nv docstring.


if __name__ == "__main__":
    unittest.main()
