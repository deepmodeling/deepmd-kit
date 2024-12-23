# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.utils.out_stat import (
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
        """Test compute_stats_property function with intensive scenario."""
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
        """Test compute_stats_property function with extensive scenario."""
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
