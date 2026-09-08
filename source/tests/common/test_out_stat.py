# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
import unittest.mock

import numpy as np

from deepmd.utils.out_stat import (
    ReduStatAccumulator,
    ReduStatScanner,
    compute_stats_do_not_distinguish_types,
    compute_stats_from_atomic,
    compute_stats_from_redu,
)


class TestOutStat(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20240227)
        ndim = 5
        nframes = 1000
        ntypes = 3
        nloc = 1000
        self.atype = rng.integers(0, ntypes, size=(nframes, nloc))
        # compute the number of atoms for each type in each frame
        self.natoms = np.zeros((nframes, ntypes), dtype=np.int64)
        for i in range(ntypes):
            self.natoms[:, i] = (self.atype == i).sum(axis=1)
        self.mean = rng.random((ntypes, ndim)) * 1e4
        self.std = rng.random((ntypes, ndim)) * 1e-3

        # generate random output
        self.output = rng.normal(
            loc=self.mean[self.atype, :],
            scale=self.std[self.atype, :],
            size=(nframes, nloc, ndim),
        )
        self.output_redu = self.output.sum(axis=1)

        return super().setUp()

    def test_compute_stats_from_redu(self) -> None:
        bias, std = compute_stats_from_redu(self.output_redu, self.natoms)
        np.testing.assert_allclose(bias, self.mean, rtol=1e-7)
        reference_std = np.array(
            [
                0.01700638138272794,
                0.01954897296228177,
                0.020281857747683162,
                0.010741237959989648,
                0.020258211828681347,
            ]
        )
        np.testing.assert_allclose(
            std,
            reference_std,
            rtol=1e-7,
        )
        # ensure the sum is close
        np.testing.assert_allclose(
            self.output_redu,
            self.natoms @ bias,
            rtol=1e-7,
        )

    def test_compute_stats_from_redu_with_assigned_bias(self) -> None:
        assigned_bias = np.full_like(self.mean, np.nan)
        assigned_bias[0] = self.mean[0]
        bias, std = compute_stats_from_redu(
            self.output_redu,
            self.natoms,
            assigned_bias=assigned_bias,
        )
        np.testing.assert_allclose(bias, self.mean, rtol=1e-7)
        np.testing.assert_allclose(bias[0], self.mean[0], rtol=1e-14)
        reference_std = np.array(
            [
                0.017015794087883902,
                0.019549011723239484,
                0.020285565914828625,
                0.01074124012073672,
                0.020283557003416414,
            ]
        )
        np.testing.assert_allclose(
            std,
            reference_std,
            rtol=1e-7,
        )
        # ensure the sum is close
        np.testing.assert_allclose(
            self.output_redu,
            self.natoms @ bias,
            rtol=1e-7,
        )

    def test_compute_stats_do_not_distinguish_types_intensive(self) -> None:
        """Test compute_stats_do_not_distinguish function with intensive scenario."""
        bias, std = compute_stats_do_not_distinguish_types(
            self.output_redu, self.natoms, intensive=True
        )
        # Test shapes
        assert bias.shape == (len(self.mean), self.output_redu.shape[1])
        assert std.shape == (len(self.mean), self.output_redu.shape[1])

        # Test values
        for fake_atom_bias in bias:
            np.testing.assert_allclose(
                fake_atom_bias, np.mean(self.output_redu, axis=0), rtol=1e-7
            )
        for fake_atom_std in std:
            np.testing.assert_allclose(
                fake_atom_std, np.std(self.output_redu, axis=0), rtol=1e-7
            )

    def test_compute_stats_do_not_distinguish_types_extensive(self) -> None:
        """Test compute_stats_do_not_distinguish function with extensive scenario."""
        bias, std = compute_stats_do_not_distinguish_types(
            self.output_redu, self.natoms
        )
        # Test shapes
        assert bias.shape == (len(self.mean), self.output_redu.shape[1])
        assert std.shape == (len(self.mean), self.output_redu.shape[1])

        # Test values
        for fake_atom_bias in bias:
            np.testing.assert_allclose(
                fake_atom_bias,
                np.array(
                    [
                        6218.91610282,
                        7183.82275736,
                        4445.23155934,
                        5748.23644722,
                        5362.8519454,
                    ]
                ),
                rtol=1e-7,
            )
        for fake_atom_std in std:
            np.testing.assert_allclose(
                fake_atom_std,
                np.array(
                    [128.78691576, 36.53743668, 105.82372405, 96.43642486, 33.68885327]
                ),
                rtol=1e-7,
            )

    def test_compute_stats_from_redu_intensive(self) -> None:
        """Test compute_stats_from_redu function with intensive scenario."""
        bias, std = compute_stats_from_redu(
            self.output_redu / self.natoms.sum(axis=1, keepdims=True),
            self.natoms,
            intensive=True,
        )
        # Test shapes
        assert bias.shape == (len(self.mean), self.output_redu.shape[1])
        assert std.shape == (self.output_redu.shape[1],)

        # Test values
        np.testing.assert_allclose(bias, self.mean, rtol=1e-6)
        reference_std = np.array(
            [
                0.00001700638138272794,
                0.00001954897296228177,
                0.000020281857747683162,
                0.000010741237959989648,
                0.000020258211828681347,
            ]
        )
        np.testing.assert_allclose(
            std,
            reference_std,
            rtol=1e-6,
        )

    def test_compute_stats_from_atomic(self) -> None:
        bias, std = compute_stats_from_atomic(self.output, self.atype)
        np.testing.assert_allclose(bias, self.mean)
        reference_std = np.array(
            [
                [
                    0.0005452949516910239,
                    0.000686732800598535,
                    0.00089423457667224,
                    7.818017989121455e-05,
                    0.0004758637035637342,
                ],
                [
                    2.0610161678825724e-05,
                    0.0007728218734771541,
                    0.0004754659308165858,
                    0.0001809007655290948,
                    0.0008187364708029638,
                ],
                [
                    0.0007935836092665254,
                    0.00031176505013516624,
                    0.0005469653430009186,
                    0.0005652240916389281,
                    0.0006087722080071852,
                ],
            ]
        )
        np.testing.assert_allclose(
            std,
            reference_std,
            rtol=1e-7,
        )


class TestReduStatAccumulator(unittest.TestCase):
    """The streaming accumulator must reproduce compute_stats_from_redu exactly."""

    def setUp(self) -> None:
        rng = np.random.default_rng(20260908)
        self.ntypes = 6
        self.ndim = 3
        nframes = 500
        self.natoms = rng.integers(1, 8, size=(nframes, self.ntypes))
        # a rare element present in a single frame, which batch sampling misses
        self.natoms[:, 3] = 0
        self.natoms[0, 3] = 1
        self.mean = rng.random((self.ntypes, self.ndim)) * 1e3
        self.output_redu = self.natoms @ self.mean + rng.normal(
            scale=1e-2, size=(nframes, self.ndim)
        )
        return super().setUp()

    def _accumulate(self, intensive: bool = False) -> ReduStatAccumulator:
        acc = ReduStatAccumulator(
            self.ntypes, self.ndim, [self.ndim], intensive=intensive
        )
        for start in range(0, self.natoms.shape[0], 7):
            acc.add(self.output_redu[start : start + 7], self.natoms[start : start + 7])
        return acc

    def test_matches_compute_stats_from_redu(self) -> None:
        for intensive in (False, True):
            with self.subTest(intensive=intensive):
                ref_bias, ref_std = compute_stats_from_redu(
                    self.output_redu, self.natoms, intensive=intensive
                )
                bias, std = self._accumulate(intensive).solve()
                np.testing.assert_allclose(bias, ref_bias, rtol=1e-9)
                np.testing.assert_allclose(std, ref_std, rtol=1e-9)

    def test_matches_with_assigned_bias(self) -> None:
        assigned_bias = np.full((self.ntypes, self.ndim), np.nan)
        assigned_bias[1] = self.mean[1]
        assigned_bias[4] = self.mean[4]
        ref_bias, ref_std = compute_stats_from_redu(
            self.output_redu.copy(),
            self.natoms.copy(),
            assigned_bias=assigned_bias,
        )
        bias, std = self._accumulate().solve(assigned_bias=assigned_bias)
        np.testing.assert_allclose(bias, ref_bias, rtol=1e-9)
        np.testing.assert_allclose(std, ref_std, rtol=1e-9)

    def test_matches_with_type_mask(self) -> None:
        type_mask = np.ones(self.ntypes, dtype=np.int64)
        type_mask[2] = 0
        ref_bias, ref_std = compute_stats_from_redu(
            self.output_redu, self.natoms * type_mask.reshape(1, -1)
        )
        bias, std = self._accumulate().solve(type_mask=type_mask)
        # the excluded type is left at the numerical zero of the min-norm solution
        np.testing.assert_allclose(bias, ref_bias, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(std, ref_std, rtol=1e-9)

    def test_repeated_compression_is_exact(self) -> None:
        ref_bias, ref_std = compute_stats_from_redu(self.output_redu, self.natoms)
        acc = ReduStatAccumulator(self.ntypes, self.ndim, [self.ndim])
        acc._compress_every = 16
        for start in range(self.natoms.shape[0]):
            acc.add(self.output_redu[start : start + 1], self.natoms[start : start + 1])
        bias, std = acc.solve()
        np.testing.assert_allclose(bias, ref_bias, rtol=1e-9)
        np.testing.assert_allclose(std, ref_std, rtol=1e-9)

    def test_counts_every_frame(self) -> None:
        acc = self._accumulate()
        self.assertEqual(acc.nframes, self.natoms.shape[0])
        np.testing.assert_array_equal(acc.natoms_total, self.natoms.sum(axis=0))

    def test_output_shape_is_preserved(self) -> None:
        rng = np.random.default_rng(0)
        acc = ReduStatAccumulator(self.ntypes, 6, [2, 3])
        acc.add(rng.random((10, 2, 3)), rng.integers(1, 5, (10, self.ntypes)))
        bias, std = acc.solve()
        self.assertEqual(bias.shape, (self.ntypes, 2, 3))
        self.assertEqual(std.shape, (2, 3))

    def test_empty_accumulator_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReduStatAccumulator(self.ntypes, self.ndim).solve()


class TestReduStatScanner(unittest.TestCase):
    def test_scan_is_performed_once_per_request(self) -> None:
        calls = []

        def scan_fn(ntypes, keys, intensive):
            calls.append((ntypes, keys, intensive))
            return unittest.mock.Mock(natoms_total=np.array([1, 0, 2]))

        scanner = ReduStatScanner(scan_fn)
        scanner.scan(3, ["energy"])
        scanner.scan(3, ["energy"])
        scanner.scan(3, ["energy"], intensive=True)
        self.assertEqual(len(calls), 2)

    def test_natoms_total_reuses_a_previous_scan(self) -> None:
        counts = np.array([1, 0, 2])
        scanner = ReduStatScanner(
            lambda ntypes, keys, intensive: unittest.mock.Mock(natoms_total=counts)
        )
        scanner.scan(3, ["energy"])
        np.testing.assert_array_equal(scanner.natoms_total(3), counts)
