# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for min_pair_dist frame filtering."""

import unittest

import numpy as np

from deepmd.dpmodel.utils.dist_check import (
    compute_min_pair_dist_single,
)


class TestComputeMinPairDistSingle(unittest.TestCase):
    """Test minimum pairwise distance computation."""

    def test_three_atoms_no_pbc(self) -> None:
        """Three atoms, closest pair is 0.3 Å."""
        coord = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.3,
                0.0,
                0.0,
            ]
        )
        atype = np.array([0, 0, 1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        np.testing.assert_almost_equal(dist, 0.3)

    def test_pbc_minimum_image(self) -> None:
        """Two atoms near opposite edges of a 10 Å cubic box.

        Real-space distance is 9.0 Å, but minimum image distance is 1.0 Å.
        """
        coord = np.array([0.5, 5.0, 5.0, 9.5, 5.0, 5.0])
        box = np.array([10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0])
        atype = np.array([0, 0])
        dist = compute_min_pair_dist_single(coord, box=box, atype=atype)
        np.testing.assert_almost_equal(dist, 1.0)

    def test_pbc_triclinic(self) -> None:
        """Triclinic box with atoms near boundary."""
        # Triclinic box: a=(10,0,0), b=(2,10,0), c=(0,0,10)
        box = np.array([10.0, 0, 0, 2.0, 10.0, 0, 0, 0, 10.0])
        coord = np.array([0.2, 0.0, 0.0, 9.8, 0.0, 0.0])
        atype = np.array([0, 0])
        dist = compute_min_pair_dist_single(coord, box=box, atype=atype)
        np.testing.assert_almost_equal(dist, 0.4, decimal=5)

    def test_virtual_atoms_excluded(self) -> None:
        """Virtual atoms (type < 0) should be excluded."""
        coord = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
            ]
        )
        atype = np.array([0, -1, 1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        np.testing.assert_almost_equal(dist, 2.0)

    def test_single_real_atom(self) -> None:
        """Only one real atom returns inf."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([0, -1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        self.assertEqual(dist, float("inf"))

    def test_all_virtual(self) -> None:
        """All virtual atoms return inf."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([-1, -1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        self.assertEqual(dist, float("inf"))

    def test_coord_shape_2d(self) -> None:
        """Accept (natoms, 3) shaped coord."""
        coord = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        atype = np.array([0, 1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        np.testing.assert_almost_equal(dist, 0.8)

    def test_stop_below_triggers_early_exit(self) -> None:
        """A pair below stop_below should still return the correct minimum."""
        coord = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 10.0, 0.0, 0.0])
        atype = np.array([0, 0, 0])
        dist = compute_min_pair_dist_single(
            coord, box=None, atype=atype, stop_below=0.1
        )
        np.testing.assert_almost_equal(dist, 0.05)

    def test_stop_below_not_triggered(self) -> None:
        """If all pairs are above stop_below, the true minimum is returned."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        atype = np.array([0, 0, 0])
        dist = compute_min_pair_dist_single(
            coord, box=None, atype=atype, stop_below=0.5
        )
        np.testing.assert_almost_equal(dist, 1.0)

    def test_multi_block_iteration(self) -> None:
        """>512 atoms exercises multiple row blocks."""
        rng = np.random.default_rng(42)
        nloc = 600
        coord = rng.uniform(0.0, 100.0, (nloc, 3))
        atype = np.zeros(nloc, dtype=np.int64)
        diff = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        ref = dist.min()

        actual = compute_min_pair_dist_single(coord, box=None, atype=atype)
        np.testing.assert_almost_equal(actual, ref, decimal=10)

    def test_coincident_atoms_zero(self) -> None:
        """Coincident real atoms should return exactly zero."""
        coord = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([0, 0, 0])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        self.assertEqual(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
