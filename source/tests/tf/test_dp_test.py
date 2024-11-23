# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil
import unittest
from pathlib import (
    Path,
)

import dpdata
import numpy as np

from deepmd.tf.entrypoints.test import test as dp_test
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

from ..infer.case import (
    get_cases,
)
from .common import (
    infer_path,
)

default_places = 6


class TestDPTest:
    def setUp(self) -> None:
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ]
        )
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])
        self.test_data = "test_dp_test"
        dpdata.System(
            data={
                "orig": np.zeros(3),
                "atom_names": ["O", "H"],
                "atom_numbs": [2, 4],
                "atom_types": np.array(self.atype),
                "cells": self.box.reshape(1, 3, 3),
                "coords": self.coords.reshape(1, 6, 3),
            }
        ).to_deepmd_npy(self.test_data)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_data, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.model_name)


class TestDPTestEner(unittest.TestCase, TestDPTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = get_cases()["se_e2_a"]
        cls.model_name = cls.case.get_model(".pb")

    def setUp(self) -> None:
        self.result = self.case.results[0]
        TestDPTest.setUp(self)
        self.expected_e = self.result.atomic_energy
        self.expected_f = self.result.force
        self.expected_v = self.result.atomic_virial

    def test_1frame(self) -> None:
        detail_file = "test_dp_test_ener_detail"
        dp_test(
            model=self.model_name,
            system=self.test_data,
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        pred_e = np.loadtxt(detail_file + ".e.out", ndmin=2)[0, 1]
        pred_f = np.loadtxt(detail_file + ".f.out", ndmin=2)[:, 3:6]
        pred_v = np.loadtxt(detail_file + ".v.out", ndmin=2)[:, 9:18]
        pred_e_peratom = np.loadtxt(detail_file + ".e_peratom.out", ndmin=2)[0, 1]
        pred_v_peratom = np.loadtxt(detail_file + ".v_peratom.out", ndmin=2)[:, 9:18]
        self.assertAlmostEqual(pred_e, np.sum(self.expected_e), places=default_places)
        np.testing.assert_almost_equal(
            pred_f, self.expected_f.reshape(-1, 3), decimal=default_places
        )
        np.testing.assert_almost_equal(
            pred_v,
            np.sum(self.expected_v.reshape(1, -1, 9), axis=1),
            decimal=default_places,
        )
        np.testing.assert_almost_equal(
            pred_e_peratom, pred_e / len(self.atype), decimal=default_places
        )
        np.testing.assert_almost_equal(
            pred_v_peratom, pred_v / len(self.atype), decimal=default_places
        )


class TestDPTestDipole(unittest.TestCase, TestDPTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "deepdipole.pb"
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deepdipole.pbtxt")), cls.model_name
        )

    def setUp(self) -> None:
        TestDPTest.setUp(self)
        self.expected_d = np.array(
            [
                -9.274180565967479195e-01,
                2.698028341272042496e00,
                2.521268387140979117e-01,
                2.927260638453461628e00,
                -8.571926301526779923e-01,
                1.667785136187720063e00,
            ]
        )
        self.expected_global_d = np.sum(self.expected_d.reshape(1, -1, 3), axis=1)
        np.save(Path(self.test_data) / "set.000" / "atomic_dipole.npy", self.expected_d)
        np.save(Path(self.test_data) / "set.000" / "dipole.npy", self.expected_global_d)

    def test_1frame(self) -> None:
        detail_file = "test_dp_test_dipole_detail"
        dp_test(
            model=self.model_name,
            system=self.test_data,
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=True,
        )
        dipole = np.loadtxt(detail_file + ".out", ndmin=2)[0, 6:12]
        np.testing.assert_almost_equal(dipole, self.expected_d, decimal=default_places)

    def test_1frame_global(self) -> None:
        detail_file = "test_dp_test_global_dipole_detail"
        dp_test(
            model=self.model_name,
            system=self.test_data,
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        dipole = np.loadtxt(detail_file + ".out", ndmin=2)[:, 3:6]
        np.testing.assert_almost_equal(
            dipole, self.expected_global_d, decimal=default_places
        )


class TestDPTestPolar(unittest.TestCase, TestDPTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "deeppolar.pb"
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deeppolar.pbtxt")), cls.model_name
        )

    def setUp(self) -> None:
        TestDPTest.setUp(self)
        self.expected_d = np.array(
            [
                1.061407927405987051e-01,
                -3.569013342133873778e-01,
                -2.862108976089940138e-02,
                -3.569013342133875444e-01,
                1.304367268874677244e00,
                1.037647501453442256e-01,
                -2.862108976089940138e-02,
                1.037647501453441284e-01,
                8.100521520762453409e-03,
                1.236797829492216616e00,
                -3.717307430531632262e-01,
                7.371515676976750919e-01,
                -3.717307430531630041e-01,
                1.127222682121889058e-01,
                -2.239181552775717510e-01,
                7.371515676976746478e-01,
                -2.239181552775717787e-01,
                4.448255365635306879e-01,
            ]
        )
        self.expected_global_d = np.sum(self.expected_d.reshape(1, -1, 9), axis=1)
        np.save(
            Path(self.test_data) / "set.000" / "atomic_polarizability.npy",
            self.expected_d,
        )
        np.save(
            Path(self.test_data) / "set.000" / "polarizability.npy",
            self.expected_global_d,
        )

    def test_1frame(self) -> None:
        detail_file = "test_dp_test_polar_detail"
        dp_test(
            model=self.model_name,
            system=self.test_data,
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=True,
        )
        polar = np.loadtxt(detail_file + ".out", ndmin=2)[0, 18:36]
        np.testing.assert_almost_equal(polar, self.expected_d, decimal=default_places)

    def test_1frame_global(self) -> None:
        detail_file = "test_dp_test_global_polar_detail"
        dp_test(
            model=self.model_name,
            system=self.test_data,
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        polar = np.loadtxt(detail_file + ".out", ndmin=2)[:, 9:18]
        np.testing.assert_almost_equal(
            polar, self.expected_global_d, decimal=default_places
        )
