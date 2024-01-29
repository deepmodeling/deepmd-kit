# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np
from deepmd.utils.pair_tab import (
    PairTab,
)


class TestPairTabPreprocessLinear(unittest.TestCase):
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

        self.tab1 = PairTab(filename=file_path, rcut=0.01)
        self.tab2 = PairTab(filename=file_path, rcut=0.02)
        self.tab3 = PairTab(filename=file_path, rcut=0.022)
        self.tab4 = PairTab(filename=file_path, rcut=0.03)
        

    def test_preprocess(self):
        
        np.testing.assert_allclose(self.tab1.vdata, np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        ))
        np.testing.assert_allclose(self.tab2.vdata, np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        ))

        # for this test case, the table does not decay to zero at rcut = 0.22,
        # in the cubic spline code, we use a fixed size grid, if will be a problem if we introduce variable gird size.
        # we will do post process to overwrite spline coefficient `a3`,`a2`,`a1`,`a0`, to ensure energy decays to `0`.
        np.testing.assert_allclose(self.tab3.vdata, np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0., 0., 0.],
            ]
        ))
        np.testing.assert_allclose(self.tab4.vdata, np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0.125, 0.2, 0.375],
                [0.03, 0., 0., 0.],
                [0.035, 0., 0., 0.],
            ]
        ))

class TestPairTabPreprocessZero(unittest.TestCase):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0., 0., 0.],
            ]
        )

        self.tab1 = PairTab(filename=file_path, rcut=0.023)
        self.tab2 = PairTab(filename=file_path, rcut=0.025)
        self.tab3 = PairTab(filename=file_path, rcut=0.028)
        

    def test_preprocess(self):
        
        np.testing.assert_allclose(self.tab1.vdata, np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0., 0., 0.],
            ]
        ))
        np.testing.assert_allclose(self.tab2.vdata, np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0., 0., 0.],
            ]
        ))
        
        np.testing.assert_allclose(self.tab3.vdata, np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
                [0.025, 0., 0., 0.],
                [0.03, 0., 0., 0.],
            ]
        ))
        np.testing.assert_equal(self.tab3.nspline,5)

        # for this test case, padding zeros between 0.025 and 0.03 will cause the cubic spline go below zero and result in negative energy values,
        # we will do post process to overwrite spline coefficient `a3`,`a2`,`a1`,`a0`, to ensure energy decays to `0`.
        temp_data = self.tab3.tab_data.reshape(2,2,-1,4)
        np.testing.assert_allclose(temp_data[:,:,-1,:], np.zeros((2,2,4)))
        