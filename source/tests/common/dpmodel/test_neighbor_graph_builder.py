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

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
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


if __name__ == "__main__":
    unittest.main()
