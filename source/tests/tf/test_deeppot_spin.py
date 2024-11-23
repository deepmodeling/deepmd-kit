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
                0.006738920830805593,
                -4.832585247816567e-06,
                -2.0276606397438153e-06,
                -4.366513387077643e-06,
                0.006741163323550941,
                1.719835794537222e-06,
                -2.1858489158912493e-06,
                1.508878980457392e-06,
                0.0067437875441185395,
                0.006736146911048136,
                -4.807779113801557e-08,
                -1.2458573644803254e-06,
                -5.393327300826365e-07,
                0.0067448511549604295,
                -3.1829145899408113e-06,
                -1.0463097199534089e-06,
                -3.109119711580798e-06,
                0.0067431538220025025,
                0.0011758392691684902,
                0.0016295934147576429,
                0.0009981689960233038,
                0.0005315778697789059,
                0.002329549082428894,
                -0.00040427012003146695,
                0.0003253969567772409,
                -0.0004043601173826382,
                0.002742369303707535,
                0.0030035100633055522,
                -0.00162379012146519,
                -0.0009953775695754462,
                -0.0005258026403329556,
                0.0018325778661179022,
                0.00040612629995906175,
                -0.00032255870528057187,
                0.0004060618020877781,
                0.0014177911924174575,
                0.013298851806357455,
                0.00905168439076821,
                0.005551741772056881,
                0.009052435097729693,
                -0.0025289322225853744,
                -0.011513486668551145,
                0.005551414024623528,
                -0.011513534014971512,
                0.009222905820377207,
                0.013299233090739378,
                0.009057905763645032,
                0.005548089545504811,
                0.009057208303608228,
                -0.0025263460059335562,
                -0.011521412196322188,
                0.005548329108520972,
                -0.011521073192743665,
                0.009220402465337748,
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
