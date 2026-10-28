# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the pair-clearance frame filter."""

import unittest

import numpy as np

from deepmd.dpmodel.utils.dist_check import (
    compute_min_pair_margin_batch,
    compute_min_pair_margin_single,
    pair_half_thresholds,
    select_frames,
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

#: A unit length scale: every pair threshold is 1 Å, so a margin is the pair
#: distance itself and the filter reduces to a plain distance floor.
UNIT_HALF = np.full(8, 0.5)


class TestComputeMinPairMarginSingle(unittest.TestCase):
    """Test minimum pair margin computation on a unit length scale."""

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
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        np.testing.assert_almost_equal(dist, 0.3)

    def test_pbc_minimum_image(self) -> None:
        """Two atoms near opposite edges of a 10 Å cubic box.

        Real-space distance is 9.0 Å, but minimum image distance is 1.0 Å.
        """
        coord = np.array([0.5, 5.0, 5.0, 9.5, 5.0, 5.0])
        box = np.array([10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0])
        atype = np.array([0, 0])
        dist = compute_min_pair_margin_single(coord, box, atype, UNIT_HALF)
        np.testing.assert_almost_equal(dist, 1.0)

    def test_pbc_triclinic(self) -> None:
        """Triclinic box with atoms near boundary."""
        # Triclinic box: a=(10,0,0), b=(2,10,0), c=(0,0,10)
        box = np.array([10.0, 0, 0, 2.0, 10.0, 0, 0, 0, 10.0])
        coord = np.array([0.2, 0.0, 0.0, 9.8, 0.0, 0.0])
        atype = np.array([0, 0])
        dist = compute_min_pair_margin_single(coord, box, atype, UNIT_HALF)
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
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        np.testing.assert_almost_equal(dist, 2.0)

    def test_single_real_atom(self) -> None:
        """Only one real atom returns inf."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([0, -1])
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        self.assertEqual(dist, float("inf"))

    def test_all_virtual(self) -> None:
        """All virtual atoms return inf."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([-1, -1])
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        self.assertEqual(dist, float("inf"))

    def test_coord_shape_2d(self) -> None:
        """Accept (natoms, 3) shaped coord."""
        coord = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        atype = np.array([0, 1])
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        np.testing.assert_almost_equal(dist, 0.8)

    def test_a_pair_inside_its_window_owns_the_margin(self) -> None:
        """A pair far inside its window is the one the scan reports."""
        coord = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 10.0, 0.0, 0.0])
        atype = np.array([0, 0, 0])
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        np.testing.assert_almost_equal(dist, 0.05)

    def test_a_pair_at_the_edge_of_its_window_reports_one(self) -> None:
        """A frame whose closest pair sits exactly on its window reads one."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        atype = np.array([0, 0, 0])
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
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

        actual = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        np.testing.assert_almost_equal(actual, ref, decimal=10)

    def test_coincident_atoms_zero(self) -> None:
        """Coincident real atoms should return exactly zero."""
        coord = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([0, 0, 0])
        dist = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        self.assertEqual(dist, 0.0)

    def test_a_screened_scan_keeps_the_side_of_the_window(self) -> None:
        """A screened scan stops at the first pair inside the window.

        It then reports the margin it had reached rather than the smallest one,
        which can only be larger and still lies below one. A frame that clears
        its window is never cut short, so its margin is the exact one.
        """
        # A lattice wide enough for several row blocks, so a scan that stops
        # early stops before reaching the closest pair.
        grid = np.arange(9) * 2.0
        coord = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1)
        coord = coord.reshape(-1, 3)
        atype = np.zeros(coord.shape[0], dtype=np.int64)
        clear = compute_min_pair_margin_single(coord, None, atype, UNIT_HALF)
        self.assertGreater(clear, 1.0)
        self.assertEqual(
            compute_min_pair_margin_single(
                coord, None, atype, UNIT_HALF, screened=True
            ),
            clear,
        )

        # One pair inside the window early in the scan, a closer one late.
        coord[1] = coord[0] + np.array([0.7, 0.0, 0.0])
        coord[721] = coord[720] + np.array([0.2, 0.0, 0.0])
        np.testing.assert_allclose(
            compute_min_pair_margin_single(coord, None, atype, UNIT_HALF),
            0.2,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            compute_min_pair_margin_single(
                coord, None, atype, UNIT_HALF, screened=True
            ),
            0.7,
            rtol=1e-12,
        )


class TestComputeMinPairMarginBatch(unittest.TestCase):
    """The batched scan gives every frame the margin of the single scan."""

    @staticmethod
    def _frames(
        nframes: int,
        natoms: int,
        seed: int,
        *,
        periodic: bool = True,
        virtual: bool = False,
        clustered: bool = False,
        sizes: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
        """Build frames of random geometry, one cell each."""
        rng = np.random.default_rng(seed)
        counts = np.full(nframes, natoms) if sizes is None else sizes
        coords, cells, types = [], [], []
        for count in counts:
            cell = np.diag([7.0, 8.5, 11.0]) + 0.6 * rng.standard_normal((3, 3))
            fractional = rng.uniform(0.0, 1.0, (count, 3))
            if clustered:
                # Two tight clusters half a cell apart: the closest pair of a
                # frame may cross the periodic boundary either way.
                fractional[: count // 2] *= 0.05
                fractional[count // 2 :] = 0.55 + 0.05 * fractional[count // 2 :]
            coord = fractional @ cell
            if count > 1 and rng.random() < 0.5:
                # Half the frames hold a pair far below any plausible bound.
                coord[1] = coord[0] + 0.05 * rng.standard_normal(3)
            atype = rng.integers(0, 4, count)
            if virtual:
                atype[rng.random(count) < 0.25] = -1
            coords.append(coord)
            cells.append(cell.reshape(9) if periodic else None)
            types.append(atype)
        boxes = np.stack(cells) if periodic else None
        return coords, boxes, types

    def _assert_matches_single(
        self, coords, boxes, types, half=UNIT_HALF, **kwargs
    ) -> None:
        expected = np.array(
            [
                compute_min_pair_margin_single(
                    coord, None if boxes is None else boxes[frame], atype, half
                )
                for frame, (coord, atype) in enumerate(zip(coords, types, strict=True))
            ]
        )
        got = compute_min_pair_margin_batch(
            np.stack(coords) if "n_node" not in kwargs else np.concatenate(coords),
            boxes,
            np.stack(types) if "n_node" not in kwargs else np.concatenate(types),
            half,
            **kwargs,
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_matches_the_single_frame_scan(self) -> None:
        for periodic in (True, False):
            for virtual in (False, True):
                for clustered in (False, True):
                    with self.subTest(
                        periodic=periodic, virtual=virtual, clustered=clustered
                    ):
                        self._assert_matches_single(
                            *self._frames(
                                17,
                                14,
                                seed=3,
                                periodic=periodic,
                                virtual=virtual,
                                clustered=clustered,
                            )
                        )

    def test_ragged_batch_matches_the_single_frame_scan(self) -> None:
        sizes = np.array([5, 12, 2, 9, 12])
        coords, boxes, types = self._frames(len(sizes), 0, seed=5, sizes=sizes)
        self._assert_matches_single(coords, boxes, types, n_node=sizes)

    def test_frames_without_a_pair_carry_infinity(self) -> None:
        coords, boxes, types = self._frames(3, 1, seed=7)
        got = compute_min_pair_margin_batch(
            np.stack(coords), boxes, np.stack(types), UNIT_HALF
        )
        np.testing.assert_array_equal(got, np.full(3, np.inf))

    def test_periodic_and_open_frames_mix_in_one_batch(self) -> None:
        """A cell of zeros marks a frame the minimum image does not apply to."""
        coords, boxes, types = self._frames(11, 9, seed=13)
        boxes = boxes.copy()
        boxes[[2, 5, 6]] = 0.0
        expected = np.array(
            [
                compute_min_pair_margin_single(
                    coord,
                    boxes[frame] if boxes[frame].any() else None,
                    atype,
                    UNIT_HALF,
                )
                for frame, (coord, atype) in enumerate(zip(coords, types, strict=True))
            ]
        )
        got = compute_min_pair_margin_batch(
            np.stack(coords), boxes, np.stack(types), UNIT_HALF
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_filler_atoms_leave_the_scan(self) -> None:
        """Padding rows share one position and must not hold up the scan.

        A rectangular batch pads its frames with atoms of a negative type at
        the origin. They carry no distance, and leaving them in the order
        would stall the window, since they never separate.
        """
        coords, boxes, types = self._frames(12, 7, seed=17)
        padded_coords, padded_types = [], []
        for coord, atype in zip(coords, types, strict=True):
            padded_coords.append(np.concatenate([coord, np.zeros((25, 3))]))
            padded_types.append(np.concatenate([atype, np.full(25, -1)]))
        expected = compute_min_pair_margin_batch(
            np.stack(coords), boxes, np.stack(types), UNIT_HALF
        )
        got = compute_min_pair_margin_batch(
            np.stack(padded_coords), boxes, np.stack(padded_types), UNIT_HALF
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_the_exact_scan_reaches_beyond_the_first_neighbours(self) -> None:
        """Without a bound the scan widens until it owns the minimum.

        The closest pair of every frame is hidden behind several atoms that
        separate it along the ordered direction, so neither the first windows
        nor a scan that screened at a distance of its own would find it.
        """
        rng = np.random.default_rng(23)
        nframes, natoms, edge = 9, 26, 14.0
        cells = np.stack([np.diag([edge, edge, edge]) for _ in range(nframes)])
        atype = np.zeros((nframes, natoms), dtype=np.int64)
        coords = np.empty((nframes, natoms, 3))
        for frame in range(nframes):
            # A row of atoms along x, spread far apart in y so that only the
            # planted pair is close in space.
            position = np.stack(
                [
                    np.linspace(1.0, 12.0, natoms),
                    rng.permutation(np.linspace(0.0, 12.0, natoms)),
                    np.zeros(natoms),
                ],
                axis=-1,
            )
            # The winning pair straddles eight of those atoms.
            position[3] = [4.0, 6.0, 0.0]
            position[12] = [4.3, 6.0, 0.0]
            coords[frame] = position
        boxes = cells.reshape(nframes, 9)
        expected = np.array(
            [
                compute_min_pair_margin_single(
                    coords[frame], boxes[frame], atype[frame], UNIT_HALF
                )
                for frame in range(nframes)
            ]
        )
        np.testing.assert_allclose(expected, 0.3, rtol=1e-12, atol=1e-12)
        # The answer is not among the pairs that neighbour each other in the
        # ordered direction, so the scan cannot stop at its first window.
        order = np.argsort(coords[:, :, 0], axis=1)
        ordered = np.take_along_axis(coords, order[:, :, np.newaxis], axis=1)
        adjacent = np.linalg.norm(ordered[:, 1:] - ordered[:, :-1], axis=-1).min(axis=1)
        self.assertTrue((adjacent > expected + 1e-9).all())
        got = compute_min_pair_margin_batch(coords, boxes, atype, UNIT_HALF)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_the_window_closes_on_the_distance_it_has_reached(self) -> None:
        """Two periodic frames on which a loose stopping rule reports too much.

        The first frame ends the scan one window early if the separation the
        window has travelled is taken twice as long as it is. The second needs
        that separation measured in the direction of the order: reading it
        through the periodic image instead lets the window close before the
        pair its other atom would have met.
        """
        edges = (6.7731, 7.9566)
        positions = (
            [
                [0.8561, 5.8872, 4.4824],
                [4.1117, 5.0969, 6.2866],
                [2.0466, 0.3791, 4.5902],
                [2.1129, 4.8734, 4.6289],
                [4.5744, 5.5052, 1.0678],
                [1.2241, 3.5232, 1.1324],
            ],
            [
                [7.6215, 2.5614, 1.8946],
                [6.9715, 6.4405, 7.6013],
                [2.2376, 1.8901, 1.6561],
                [7.7785, 7.3701, 4.1730],
                [2.2114, 4.5956, 4.2626],
                [4.6901, 5.2300, 3.8674],
                [0.3614, 6.0416, 7.4999],
                [0.1976, 4.1174, 2.7984],
            ],
        )
        for edge, coord in zip(edges, positions, strict=True):
            with self.subTest(edge=edge):
                frame = np.array(coord)
                box = np.diag([edge, edge, edge]).reshape(1, 9)
                atype = np.zeros((1, frame.shape[0]), dtype=np.int64)
                expected = compute_min_pair_margin_single(
                    frame, box[0], atype[0], UNIT_HALF
                )
                got = compute_min_pair_margin_batch(
                    frame[np.newaxis], box, atype, UNIT_HALF
                )
                np.testing.assert_allclose(got, [expected], rtol=1e-12, atol=1e-12)

    def test_a_screening_bound_keeps_the_side_of_every_frame(self) -> None:
        """A bounded scan keeps the margin of every frame that fails.

        Bounding the search leaves a frame that clears its window with a
        margin it reaches rather than its minimum, so the value may grow; a
        frame that fails keeps the exact margin, since the bound is then no
        longer the smaller of the two.
        """
        for half in (UNIT_HALF, np.array([0.18, 0.31, 0.46, 0.75, 0.2, 0.2, 0.2, 0.2])):
            with self.subTest(half=half[0]):
                coords, boxes, types = self._frames(23, 16, seed=11)
                exact = compute_min_pair_margin_batch(
                    np.stack(coords), boxes, np.stack(types), half
                )
                screened = compute_min_pair_margin_batch(
                    np.stack(coords), boxes, np.stack(types), half, screened=True
                )
                self.assertTrue((exact < 1.0).any() and (exact >= 1.0).any())
                np.testing.assert_array_equal(screened < 1.0, exact < 1.0)
                np.testing.assert_allclose(
                    screened[exact < 1.0], exact[exact < 1.0], rtol=1e-12, atol=1e-12
                )
                self.assertTrue((screened >= exact - 1e-12).all())


class TestPairLengthScale(unittest.TestCase):
    """The window each pair carries is the sum of its two half-thresholds."""

    def test_absolute_scale_is_a_plain_distance_floor(self) -> None:
        """Every element gets the unit radius.

        A pair's threshold is then the fraction itself and the margin is the
        pair distance divided by it.
        """
        half = pair_half_thresholds(0.5, "absolute", ntypes=3)
        np.testing.assert_allclose(half, 0.25)
        coord = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0])
        atype = np.array([0, 2])
        margin = compute_min_pair_margin_single(coord, None, atype, half)
        np.testing.assert_allclose(margin, 0.8 / 0.5)

    def test_covalent_scale_sizes_each_pair_separately(self) -> None:
        """Two pairs at one distance carry different margins."""
        half = np.array([0.1, 0.4])
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        close = compute_min_pair_margin_single(coord, None, np.array([1, 1]), half)
        wide = compute_min_pair_margin_single(coord, None, np.array([0, 0]), half)
        np.testing.assert_allclose(close, 1.0 / 0.8)
        np.testing.assert_allclose(wide, 1.0 / 0.2)

    def test_the_minimum_margin_pair_is_not_the_minimum_distance_pair(self) -> None:
        """The scan tracks the margin, which a closer pair need not own.

        The last two atoms are the closest pair of the frame, but they are
        small and their window is narrow, so they clear it comfortably. The
        wider-apart pair of large atoms sits deeper inside its own window and
        owns the margin.
        """
        half = np.array([0.05, 1.0])
        coord = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 6.0, 0.0, 0.0, 6.4, 0.0, 0.0])
        atype = np.array([1, 1, 0, 0])
        distances = np.array([1.5, 0.4])
        margins = distances / np.array([2.0, 0.1])
        self.assertLess(distances[1], distances[0])
        self.assertLess(margins[0], margins[1])
        got = compute_min_pair_margin_single(coord, None, atype, half)
        np.testing.assert_allclose(got, margins[0])
        batched = compute_min_pair_margin_batch(
            coord[np.newaxis], None, atype[np.newaxis], half
        )
        np.testing.assert_allclose(batched, [margins[0]])

    def test_batch_matches_single_under_a_per_pair_scale(self) -> None:
        """The batched scan owns the margin for element-dependent windows."""
        half = np.array([0.18, 0.31, 0.46, 0.75, 0.2, 0.2, 0.2, 0.2])
        for clustered in (False, True):
            with self.subTest(clustered=clustered):
                coords, boxes, types = TestComputeMinPairMarginBatch._frames(
                    19, 15, seed=29, clustered=clustered
                )
                expected = np.array(
                    [
                        compute_min_pair_margin_single(coord, boxes[frame], atype, half)
                        for frame, (coord, atype) in enumerate(
                            zip(coords, types, strict=True)
                        )
                    ]
                )
                got = compute_min_pair_margin_batch(
                    np.stack(coords), boxes, np.stack(types), half
                )
                np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_covalent_scale_needs_a_type_map(self) -> None:
        with self.assertRaisesRegex(ValueError, "type_map"):
            pair_half_thresholds(0.26, "covalent", ntypes=3)

    def test_covalent_scale_reads_the_periodic_table(self) -> None:
        half = pair_half_thresholds(0.5, "covalent", type_map=["H", "O"])
        # Half of the Pyykko single-bond radii of hydrogen and oxygen.
        np.testing.assert_allclose(half, [0.16, 0.315])

    def test_unknown_scale_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "length scale"):
            pair_half_thresholds(0.26, "vdw", ntypes=3)


class TestSelectFrames(unittest.TestCase):
    """`select_frames` slices the frame axis and, when ragged, the node axis."""

    def test_rectangular_batch(self) -> None:
        batch = {
            "coord": np.arange(3 * 6, dtype=np.float64).reshape(3, 6),
            "energy": np.array([[1.0], [2.0], [3.0]]),
            "find_energy": np.bool_(True),
            "box": None,
        }
        kept = select_frames(batch, np.array([True, False, True]))
        np.testing.assert_array_equal(kept["coord"], batch["coord"][[0, 2]])
        np.testing.assert_array_equal(kept["energy"][:, 0], [1.0, 3.0])
        self.assertTrue(kept["find_energy"])
        self.assertIsNone(kept["box"])

    def test_ragged_batch(self) -> None:
        # Three frames of two, one and three atoms on one flat node axis.
        n_node = np.array([2, 1, 3])
        frame_of_node = np.repeat(np.arange(3), n_node)
        batch = {
            "n_node": n_node,
            "atype": frame_of_node.copy(),
            "coord": np.repeat(frame_of_node, 3).reshape(6, 3).astype(np.float64),
            # A per-atom field stored flat, with two entries per atom.
            "aparam": np.repeat(frame_of_node, 2).astype(np.float64),
            "energy": np.array([[10.0], [11.0], [12.0]]),
            "box": np.arange(27, dtype=np.float64).reshape(3, 9),
        }
        kept = select_frames(batch, np.array([True, False, True]))
        np.testing.assert_array_equal(kept["n_node"], [2, 3])
        np.testing.assert_array_equal(kept["atype"], [0, 0, 2, 2, 2])
        np.testing.assert_array_equal(kept["coord"][:, 0], [0, 0, 2, 2, 2])
        np.testing.assert_array_equal(kept["aparam"], np.repeat([0, 0, 2, 2, 2], 2))
        np.testing.assert_array_equal(kept["energy"][:, 0], [10.0, 12.0])
        np.testing.assert_array_equal(kept["box"], batch["box"][[0, 2]])

    def test_ragged_batch_of_single_atoms(self) -> None:
        # One atom per frame makes the frame axis and the node axis coincide.
        batch = {
            "n_node": np.ones(3, dtype=np.int64),
            "atype": np.array([0, 1, 2]),
            "energy": np.array([[10.0], [11.0], [12.0]]),
        }
        kept = select_frames(batch, np.array([False, True, True]))
        np.testing.assert_array_equal(kept["atype"], [1, 2])
        np.testing.assert_array_equal(kept["energy"][:, 0], [11.0, 12.0])
        np.testing.assert_array_equal(kept["n_node"], [1, 1])


class _CyclicDataSystem:
    """Data source of one-frame batches that cycle through each system.

    Frame ``k`` of a system is tagged by ``coord == k`` and carries the listed
    pair margin, so a test reads off which frames were kept.
    """

    def __init__(self, margins: list[list[float]], derived: bool = True) -> None:
        self.margins = margins
        self.derived = derived
        self.cursor = [0] * len(margins)

    def get_nsystems(self) -> int:
        return len(self.margins)

    def get_nbatches(self) -> list[int]:
        return [len(system) for system in self.margins]

    def get_batch(self, sys_idx: int) -> dict:
        frame = self.cursor[sys_idx] % len(self.margins[sys_idx])
        self.cursor[sys_idx] += 1
        return {
            "coord": np.full((1, 6), float(frame)),
            "type": np.zeros((1, 2), dtype=np.int32),
            "natoms_vec": np.array([2, 2, 2]),
            "pair_margin": np.array([[self.margins[sys_idx][frame]]]),
            "find_pair_margin": float(self.derived),
        }


class TestStatisticsFilter(unittest.TestCase):
    """`make_stat_input` packs the frames whose pairs clear their window."""

    def test_scan_replaces_filtered_batches(self) -> None:
        data = _CyclicDataSystem([[0.2, 0.3, 1.5, 0.1, 1.2], [0.1, 0.2, 0.3]])
        packed = make_stat_input(data, nbatches=2, min_pair_dist=1.0)
        # The second system holds no valid frame and is skipped after one
        # pass; the first one yields its two valid frames.
        self.assertEqual(len(packed), 1)
        np.testing.assert_array_equal(packed[0]["coord"][:, 0], [2.0, 4.0])
        np.testing.assert_array_equal(packed[0]["pair_margin"][:, 0], [1.5, 1.2])
        self.assertEqual(data.cursor, [5, 3])

    def test_scan_stops_at_the_requested_batches(self) -> None:
        data = _CyclicDataSystem([[1.5, 0.2, 1.2, 1.4]])
        packed = make_stat_input(data, nbatches=2, min_pair_dist=1.0)
        np.testing.assert_array_equal(packed[0]["coord"][:, 0], [0.0, 2.0])
        self.assertEqual(data.cursor, [3])

    def test_disabled_filter_takes_the_next_batches(self) -> None:
        data = _CyclicDataSystem([[0.2, 0.3, 1.5], [0.1, 0.2]])
        packed = make_stat_input(data, nbatches=3)
        np.testing.assert_array_equal(packed[0]["coord"][:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(packed[1]["coord"][:, 0], [0.0, 1.0, 0.0])

    def test_dataset_without_valid_frames_is_reported(self) -> None:
        data = _CyclicDataSystem([[0.2, 0.3], [0.1]])
        with self.assertRaisesRegex(RuntimeError, "beyond the filter radius"):
            make_stat_input(data, nbatches=2, min_pair_dist=1.0)

    def test_underived_distance_is_refused(self) -> None:
        data = _CyclicDataSystem([[1.5, 1.2]], derived=False)
        with self.assertRaisesRegex(RuntimeError, "did not derive"):
            make_stat_input(data, nbatches=1, min_pair_dist=1.0)


if __name__ == "__main__":
    unittest.main()
