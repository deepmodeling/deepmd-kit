# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest
from unittest.mock import (
    patch,
)

import numpy as np

from deepmd.utils.pair_tab import (
    PairTab,
)


class TestPairTabPreprocessExtrapolate(unittest.TestCase):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        )

        self.tab1 = PairTab(filename=file_path, rcut=0.028)
        self.tab2 = PairTab(filename=file_path, rcut=0.02)
        self.tab3 = PairTab(filename=file_path, rcut=0.022)
        self.tab4 = PairTab(filename=file_path, rcut=0.03)
        self.tab5 = PairTab(filename=file_path, rcut=0.032)

    def test_deserialize(self) -> None:
        deserialized_tab = PairTab.deserialize(self.tab1.serialize())
        np.testing.assert_allclose(self.tab1.vdata, deserialized_tab.vdata)
        np.testing.assert_allclose(self.tab1.rmin, deserialized_tab.rmin)
        np.testing.assert_allclose(self.tab1.rmax, deserialized_tab.rmax)
        np.testing.assert_allclose(self.tab1.hh, deserialized_tab.hh)
        np.testing.assert_allclose(self.tab1.ntypes, deserialized_tab.ntypes)
        np.testing.assert_allclose(self.tab1.rcut, deserialized_tab.rcut)
        np.testing.assert_allclose(self.tab1.nspline, deserialized_tab.nspline)
        np.testing.assert_allclose(self.tab1.tab_info, deserialized_tab.tab_info)
        np.testing.assert_allclose(self.tab1.tab_data, deserialized_tab.tab_data)

    def test_preprocess(self) -> None:
        np.testing.assert_allclose(
            self.tab1.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.0, 0.0],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )
        np.testing.assert_allclose(
            self.tab2.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )

        # for this test case, the table does not decay to zero at rcut = 0.22,
        # in the cubic spline code, we use a fixed size grid, if will be a problem if we introduce variable grid size.
        # we will do post process to overwrite spline coefficient `a3`,`a2`,`a1`,`a0`, to ensure energy decays to `0`.
        np.testing.assert_allclose(
            self.tab3.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.0, 0.0],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )

        np.testing.assert_allclose(
            self.tab4.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.0, 0.0],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )

        np.testing.assert_allclose(
            self.tab5.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.12468, 0.1992, 0.3741],
                    [0.03, 0.0, 0.0, 0.0],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )


class TestPairTabPreprocessZero(unittest.TestCase):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0.0, 0.0, 0.0],
            ]
        )

        self.tab1 = PairTab(filename=file_path, rcut=0.023)
        self.tab2 = PairTab(filename=file_path, rcut=0.025)
        self.tab3 = PairTab(filename=file_path, rcut=0.028)
        self.tab4 = PairTab(filename=file_path, rcut=0.033)

    def test_preprocess(self) -> None:
        np.testing.assert_allclose(
            self.tab1.vdata,
            np.array(
                [
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.0, 0.0],
                ]
            ),
        )
        np.testing.assert_allclose(
            self.tab2.vdata,
            np.array(
                [
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.0, 0.0],
                ]
            ),
        )

        np.testing.assert_allclose(
            self.tab3.vdata,
            np.array(
                [
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.0, 0.0],
                    [0.03, 0.0, 0.0, 0.0],
                ]
            ),
        )

        np.testing.assert_allclose(
            self.tab4.vdata,
            np.array(
                [
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.0, 0.0],
                    [0.03, 0.0, 0.0, 0.0],
                    [0.035, 0.0, 0.0, 0.0],
                ]
            ),
        )


class TestPairTabPreprocessUneven(unittest.TestCase):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0.0, 0.1, 0.0],
            ]
        )

        self.tab1 = PairTab(filename=file_path, rcut=0.025)
        self.tab2 = PairTab(filename=file_path, rcut=0.028)
        self.tab3 = PairTab(filename=file_path, rcut=0.03)
        self.tab4 = PairTab(filename=file_path, rcut=0.037)

    def test_preprocess(self) -> None:
        np.testing.assert_allclose(
            self.tab1.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.1, 0.0],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )
        np.testing.assert_allclose(
            self.tab2.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.1, 0.0],
                    [0.03, 0.0, 0.0, 0.0],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )

        np.testing.assert_allclose(
            self.tab3.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.1, 0.0],
                    [0.03, 0.0, 0.0, 0.0],
                ]
            ),
            rtol=1e-04,
            atol=1e-04,
        )

        np.testing.assert_allclose(
            self.tab4.vdata,
            np.array(
                [
                    [0.005, 1.0, 2.0, 3.0],
                    [0.01, 0.8, 1.6, 2.4],
                    [0.015, 0.5, 1.0, 1.5],
                    [0.02, 0.25, 0.4, 0.75],
                    [0.025, 0.0, 0.1, 0.0],
                    [0.03, 0.0, 0.04963, 0.0],
                    [0.035, 0.0, 0.0, 0.0],
                ]
            ),
            rtol=1e-03,
            atol=1e-03,
        )


class TestPairTabGridSpacing(unittest.TestCase):
    @patch("numpy.loadtxt")
    def test_non_uniform_grid(self, mock_loadtxt) -> None:
        mock_loadtxt.return_value = np.array(
            [
                [0.00, 1.0],
                [0.01, 0.8],
                [0.02, 0.6],
                [0.09, 0.3],
                [0.16, 0.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "evenly spaced"):
            PairTab(filename="dummy_path", rcut=0.16)

    @patch("numpy.loadtxt")
    def test_duplicate_distances(self, mock_loadtxt) -> None:
        # a constant zero stride passes the uniformity check but yields hh == 0
        mock_loadtxt.return_value = np.array(
            [
                [0.01, 1.0],
                [0.01, 0.8],
                [0.01, 0.6],
                [0.01, 0.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PairTab(filename="dummy_path", rcut=0.04)

    @patch("numpy.loadtxt")
    def test_descending_grid(self, mock_loadtxt) -> None:
        # a constant negative stride passes the uniformity check but yields hh < 0
        mock_loadtxt.return_value = np.array(
            [
                [0.04, 1.0],
                [0.03, 0.8],
                [0.02, 0.6],
                [0.01, 0.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PairTab(filename="dummy_path", rcut=0.04)

    @patch("numpy.loadtxt")
    def test_failed_reinit_keeps_state(self, mock_loadtxt) -> None:
        uniform = np.array(
            [
                [0.00, 1.0],
                [0.01, 0.8],
                [0.02, 0.6],
                [0.03, 0.3],
                [0.04, 0.0],
            ]
        )
        mock_loadtxt.return_value = uniform
        tab = PairTab(filename="dummy_path", rcut=0.04)
        # serialize() hands back the live arrays, so snapshot them
        expected = copy.deepcopy(tab.serialize())

        mock_loadtxt.return_value = np.array(
            [
                [0.00, 1.0],
                [0.01, 0.8],
                [0.02, 0.6],
                [0.09, 0.3],
                [0.16, 0.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "evenly spaced"):
            tab.reinit(filename="dummy_path", rcut=0.16)

        actual = tab.serialize()
        for key in ("rmin", "rmax", "hh", "ntypes", "rcut", "nspline"):
            self.assertEqual(actual[key], expected[key])
        for key in ("vdata", "tab_info", "tab_data"):
            np.testing.assert_array_equal(
                actual["@variables"][key], expected["@variables"][key]
            )

    @patch("numpy.loadtxt")
    def test_uniform_grid(self, mock_loadtxt) -> None:
        mock_loadtxt.return_value = np.array(
            [
                [0.00, 1.0],
                [0.01, 0.8],
                [0.02, 0.6],
                [0.03, 0.3],
                [0.04, 0.0],
            ]
        )
        tab = PairTab(filename="dummy_path", rcut=0.04)
        np.testing.assert_allclose(tab.hh, 0.01)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
