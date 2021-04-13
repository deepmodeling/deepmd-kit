import os,sys,platform,shutil,dpdata
import numpy as np
import unittest

from infer.convert2pb import convert_pbtxt_to_pb
from deepmd.infer import DeepDipole
from common import tests_path

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    default_places = 4
else :
    default_places = 10

class TestDeepDipolePBC(unittest.TestCase) :
    def setUp(self):
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deepdipole.pbtxt")), "deepdipole.pb")
        self.dp = DeepDipole("deepdipole.pb")
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])
        self.expected_d = np.array([-9.274180565967479195e-01,2.698028341272042496e+00,2.521268387140979117e-01,2.927260638453461628e+00,-8.571926301526779923e-01,1.667785136187720063e+00])

    def tearDown(self):
        os.remove("deepdipole.pb")    

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 4.0, places = default_places)
        self.assertEqual(self.dp.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp.get_sel_type(), [0])

    def test_1frame_atm(self):
        dd = self.dp.eval(self.coords, self.box, self.atype)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(dd.shape, (nframes,nsel,3))
        # check values
        for ii in range(dd.size):
            self.assertAlmostEqual(dd.reshape([-1])[ii], self.expected_d.reshape([-1])[ii], places = default_places)

    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        dd = self.dp.eval(coords2, box2, self.atype)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        nsel = 2
        self.assertEqual(dd.shape, (nframes,nsel,3))
        # check values
        expected_d = np.concatenate((self.expected_d, self.expected_d))
        for ii in range(dd.size):
            self.assertAlmostEqual(dd.reshape([-1])[ii], expected_d.reshape([-1])[ii], places = default_places)


