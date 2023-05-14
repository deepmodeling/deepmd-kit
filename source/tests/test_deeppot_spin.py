import os
import unittest

import numpy as np
from common import (
    tests_path,
)

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.utils.convert import (
    convert_pbtxt_to_pb,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


class TestDeepPotAPBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(tests_path / os.path.join("infer", "deepspin.pbtxt")), "deepspin.pb"
        )
        cls.dp = DeepPot("deepspin.pb")

    def setUp(self):
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
                -0.00915789768474276,
                -0.024102013468360273,
                -0.01481967112707605,
                0.009135257767653032,
                0.024160658997748663,
                0.01482057539389492,
                -1.2706659303925863e-05,
                4.204708035318662e-05,
                -3.0069716814263983e-06,
                3.5346576393846386e-05,
                -0.00010069260974085787,
                2.1027048623316586e-06,
            ]
        )
        self.expected_fm = np.array(
            [
                -0.0694091053797398,
                0.1584293276543392,
                0.09706120404438334,
                0.06940793500220498,
                -0.1584235372080326,
                -0.09706642160777694,
            ]
        )
        self.expected_e = np.array(
            [
                -6.58889686395335,
                -6.588893275795169,
                -4.199849436761137,
                -4.200132571189533,
            ]
        )
        self.expected_v = np.array(
            [
                -0.05704674056991243,
                4.510979995814183e-05,
                1.3270434109576416e-05,
                4.000829297618676e-05,
                -0.05706160346108191,
                -4.2556568048610295e-06,
                1.2516013011240753e-05,
                -1.5009094169087907e-06,
                -0.05708822317008247,
                -0.05701483689301932,
                -1.028688864002346e-05,
                9.632611979899704e-06,
                -4.564767367100731e-06,
                -0.05709611822648184,
                1.709158858780274e-05,
                9.293420715471366e-06,
                1.797222324711753e-05,
                -0.05708518039162609,
                -0.04562926339490811,
                0.013565542733630281,
                0.008308027692994272,
                0.007930968503845335,
                -0.03346887728226176,
                -0.003277450038705189,
                0.004855290948551196,
                -0.0032764325341609933,
                -0.030128952888001017,
                -0.02721345777367022,
                -0.013546977028072798,
                -0.008301954884923411,
                -0.007912035139894842,
                -0.03943924378934477,
                0.0032811286324871698,
                -0.004849899201136112,
                0.003282386565809049,
                -0.042787508101382925,
                0.12974926347798446,
                -0.1232084006347027,
                -0.07549620740015538,
                -0.1232083276739148,
                0.2765216045700461,
                0.12855029170069582,
                -0.07549466146406651,
                0.1285432569093067,
                0.1453453938026923,
                0.12976207341985507,
                -0.12322113560020188,
                -0.07550155697475862,
                -0.12322219683367369,
                0.27653830690058406,
                0.1285560167288026,
                -0.07550132823782887,
                0.12855714070027832,
                0.1453567617513088,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        os.remove("deepspin.pb")
        cls.dp = None

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 3)
        self.assertEqual(self.dp.get_ntypes_spin(), 1)
        self.assertAlmostEqual(self.dp.get_rcut(), 5.6, places=default_places)
        self.assertEqual(self.dp.get_type_map(), ["Ni", "O"])
        self.assertEqual(self.dp.get_dim_fparam(), 0)
        self.assertEqual(self.dp.get_dim_aparam(), 0)

    def test_1frame(self):
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

    def test_1frame_atm(self):
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

    def test_2frame_atm(self):
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
        print(force_r.shape, "asfasfas")
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
