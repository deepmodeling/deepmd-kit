# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.tf.infer import (
    DeepPot,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

from .common import (
    infer_path,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


class TestDeepPotAPBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deepspin.pbtxt")), "deepspin.pb"
        )
        cls.dp = DeepPot("deepspin.pb")

    def setUp(self) -> None:
        self.coords = np.array(
            [
                0.0,
                0.0,
                0.0,
                6.82897034,
                2.06108769,
                1.26254448,
                3.41448517,
                1.03054385,
                0.63127224,
                10.2434555,
                3.09163154,
                1.89381672,
                -0.13345306,
                0.3215266,
                0.19700489,
                6.9624234,
                1.73956109,
                1.06553959,
            ]
        )
        self.atype = [0, 0, 1, 1, 2, 2]
        self.box = np.array(
            [
                5.123070480591589,
                -0.0004899308730513,
                -0.0001342698153832,
                4.267692376433796,
                2.834040245862882,
                0.0002085210138428,
                4.267638127861819,
                1.28703347467357,
                2.525122646760162,
            ]
        )
        self.expected_fr = np.array(
            [
                0.0009702072190042321,
                0.0013075955964389485,
                0.000797268590135812,
                -0.0009702845866401896,
                -0.0013004371103835927,
                -0.0008016334141859385,
                -1.85920844170827e-07,
                6.756200201089198e-07,
                -4.7165306210078025e-08,
                2.632884801233169e-07,
                -7.834106075296872e-06,
                4.411989356340548e-06,
            ]
        )
        self.expected_fm = np.array(
            [
                0.012082279998520568,
                -0.028390034321450108,
                -0.01739567760524213,
                -0.012082002187747073,
                0.028391624267302143,
                0.017394701366427952,
            ]
        )
        self.expected_e = np.array(
            [
                -7.340049376708785,
                -7.340049621809135,
                -4.242426882460533,
                -4.242704500680997,
            ]
        )
        self.expected_v = np.array(
            [
                6.7389218100709480e-03,
                -4.8325856380477971e-06,
                -2.0276608489989546e-06,
                -4.3665137929832525e-06,
                6.7411643036993114e-03,
                1.7198356948053967e-06,
                -2.1858491270862461e-06,
                1.5088788830234230e-06,
                6.7437885244728919e-03,
                6.7361478902413117e-03,
                -4.8078100635293677e-08,
                -1.2458575794766199e-06,
                -5.3933302391392304e-07,
                6.7448521352015227e-03,
                -3.1829147673636325e-06,
                -1.0463099330028346e-06,
                -3.1091198913203426e-06,
                6.7431548023373393e-03,
                1.1758393709423701e-03,
                1.6295934147249050e-03,
                9.9816899600400997e-04,
                5.3157786973548479e-04,
                2.3295491843105549e-03,
                -4.0427012004858785e-04,
                3.2539695675140221e-04,
                -4.0436011739976345e-04,
                2.7423694056090396e-03,
                3.0035101651579260e-03,
                -1.6237901215193358e-03,
                -9.9537756960785912e-04,
                -5.2580264037641934e-04,
                1.8325779681099199e-03,
                4.0612629994049175e-04,
                -3.2255870530643519e-04,
                4.0606180206920531e-04,
                1.4177912944308042e-03,
                1.8433191452566621e-02,
                -3.3184085163746957e-03,
                -2.0276273764762254e-03,
                -3.0118505453589745e-03,
                2.6537383325897366e-02,
                6.2959458489708067e-03,
                -1.8408427514310752e-03,
                6.2965273489888063e-03,
                2.0135437472640596e-02,
                1.8433454681852444e-02,
                -3.3119027150549205e-03,
                -2.0311053287539139e-03,
                -3.0077529836584696e-03,
                2.6541597362584303e-02,
                6.2890177146544546e-03,
                -1.8435128168778851e-03,
                6.2879886773803640e-03,
                2.0132321710492562e-02,
            ]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove("deepspin.pb")
        cls.dp = None

    def test_attrs(self) -> None:
        self.assertEqual(self.dp.get_ntypes(), 3)
        self.assertEqual(self.dp.get_ntypes_spin(), 1)
        self.assertAlmostEqual(self.dp.get_rcut(), 5.6, places=default_places)
        self.assertEqual(self.dp.get_type_map(), ["Ni", "O"])
        self.assertEqual(self.dp.get_dim_fparam(), 0)
        self.assertEqual(self.dp.get_dim_aparam(), 0)

    def test_1frame(self) -> None:
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic=False)

        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        # check values
        force = ff.reshape([1, -1])
        force_r = np.split(force, indices_or_sections=[12, 18], axis=1)[0]
        force_m = np.split(force, indices_or_sections=[12, 18], axis=1)[1]
        np.testing.assert_almost_equal(
            force_r.ravel(), self.expected_fr.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            force_m.ravel(), self.expected_fm.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_1frame_atm(self) -> None:
        ee, ff, vv, ae, av = self.dp.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, 6, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, 4, 1))
        self.assertEqual(av.shape, (nframes, 6, 9))
        # check values
        force = ff.reshape([1, -1])
        force_r = np.split(force, indices_or_sections=[12, 18], axis=1)[0]
        force_m = np.split(force, indices_or_sections=[12, 18], axis=1)[1]
        np.testing.assert_almost_equal(
            force_r.ravel(), self.expected_fr.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            force_m.ravel(), self.expected_fm.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            ae.ravel(), self.expected_e.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            av.ravel(), self.expected_v.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_2frame_atm(self) -> None:
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee, ff, vv, ae, av = self.dp.eval(coords2, box2, self.atype, atomic=True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, 4, 1))
        self.assertEqual(av.shape, (nframes, natoms, 9))
        # check values
        force = ff.reshape([2, -1])
        force_r = np.split(force, indices_or_sections=[12, 18], axis=1)[0]
        force_m = np.split(force, indices_or_sections=[12, 18], axis=1)[1]
        expected_fr = np.concatenate((self.expected_fr, self.expected_fr), axis=0)
        expected_fm = np.concatenate((self.expected_fm, self.expected_fm), axis=0)
        expected_e = np.concatenate((self.expected_e, self.expected_e), axis=0)
        expected_v = np.concatenate((self.expected_v, self.expected_v), axis=0)
        np.testing.assert_almost_equal(
            force_r.ravel(), expected_fr.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            force_m.ravel(), expected_fm.ravel(), default_places
        )
        np.testing.assert_almost_equal(ae.ravel(), expected_e.ravel(), default_places)
        np.testing.assert_almost_equal(av.ravel(), expected_v.ravel(), default_places)
        expected_se = np.sum(expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)
