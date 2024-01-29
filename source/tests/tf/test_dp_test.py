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

from .common import (
    infer_path,
)

default_places = 6


class TestDPTest:
    def setUp(self):
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

    def tearDown(self):
        shutil.rmtree(self.test_data, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.model_name)


class TestDPTestEner(unittest.TestCase, TestDPTest):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "deeppot.pb"
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deeppot.pbtxt")), cls.model_name
        )

    def setUp(self):
        TestDPTest.setUp(self)
        self.expected_e = np.array(
            [
                -9.275780747115504710e01,
                -1.863501786584258468e02,
                -1.863392472863538103e02,
                -9.279281325486221021e01,
                -1.863671545232153903e02,
                -1.863619822847602165e02,
            ]
        )
        self.expected_f = np.array(
            [
                -3.034045420701179663e-01,
                8.405844663871177014e-01,
                7.696947487118485642e-02,
                7.662001266663505117e-01,
                -1.880601391333554251e-01,
                -6.183333871091722944e-01,
                -5.036172391059643427e-01,
                -6.529525836149027151e-01,
                5.432962643022043459e-01,
                6.382357912332115024e-01,
                -1.748518296794561167e-01,
                3.457363524891907125e-01,
                1.286482986991941552e-03,
                3.757251165286925043e-01,
                -5.972588700887541124e-01,
                -5.987006197104716154e-01,
                -2.004450304880958100e-01,
                2.495901655353461868e-01,
            ]
        )
        self.expected_v = np.array(
            [
                -2.912234126853306959e-01,
                -3.800610846612756388e-02,
                2.776624987489437202e-01,
                -5.053761003913598976e-02,
                -3.152373041953385746e-01,
                1.060894290092162379e-01,
                2.826389131596073745e-01,
                1.039129970665329250e-01,
                -2.584378792325942586e-01,
                -3.121722367954994914e-01,
                8.483275876786681990e-02,
                2.524662342344257682e-01,
                4.142176771106586414e-02,
                -3.820285230785245428e-02,
                -2.727311173065460545e-02,
                2.668859789777112135e-01,
                -6.448243569420382404e-02,
                -2.121731470426218846e-01,
                -8.624335220278558922e-02,
                -1.809695356746038597e-01,
                1.529875294531883312e-01,
                -1.283658185172031341e-01,
                -1.992682279795223999e-01,
                1.409924999632362341e-01,
                1.398322735274434292e-01,
                1.804318474574856390e-01,
                -1.470309318999652726e-01,
                -2.593983661598450730e-01,
                -4.236536279233147489e-02,
                3.386387920184946720e-02,
                -4.174017537818433543e-02,
                -1.003500282164128260e-01,
                1.525690815194478966e-01,
                3.398976109910181037e-02,
                1.522253908435125536e-01,
                -2.349125581341701963e-01,
                9.515545977581392825e-04,
                -1.643218849228543846e-02,
                1.993234765412972564e-02,
                6.027265332209678569e-04,
                -9.563256398907417355e-02,
                1.510815124001868293e-01,
                -7.738094816888557714e-03,
                1.502832772532304295e-01,
                -2.380965783745832010e-01,
                -2.309456719810296654e-01,
                -6.666961081213038098e-02,
                7.955566551234216632e-02,
                -8.099093777937517447e-02,
                -3.386641099800401927e-02,
                4.447884755740908608e-02,
                1.008593228579038742e-01,
                4.556718179228393811e-02,
                -6.078081273849572641e-02,
            ]
        )

    def test_1frame(self):
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
    def setUpClass(cls):
        cls.model_name = "deepdipole.pb"
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deepdipole.pbtxt")), cls.model_name
        )

    def setUp(self):
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

    def test_1frame(self):
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

    def test_1frame_global(self):
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
    def setUpClass(cls):
        cls.model_name = "deeppolar.pb"
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deeppolar.pbtxt")), cls.model_name
        )

    def setUp(self):
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

    def test_1frame(self):
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

    def test_1frame_global(self):
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
