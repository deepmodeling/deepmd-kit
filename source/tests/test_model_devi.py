from deepmd.infer import DeepPotential
import unittest
import os, sys, shutil
import numpy as np
from deepmd.infer import calc_model_devi
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from infer.convert2pb import convert_pbtxt_to_pb
from common import gen_data, tests_path


class TestMakeModelDevi(unittest.TestCase):
    def setUp(self):
        gen_data()
        self.data_dir = "system"
        self.pbtxts = [os.path.join(tests_path, "infer/deeppot.pbtxt"),
                       os.path.join(tests_path, "infer/deeppot-1.pbtxt")]
        self.graph_dirs = [pbtxt.replace("pbtxt", "pb") for pbtxt in self.pbtxts]
        for pbtxt, pb in zip(self.pbtxts, self.graph_dirs):
            convert_pbtxt_to_pb(pbtxt, pb)
        self.graphs = [DeepPotential(pb) for pb in self.graph_dirs]
        self.output = os.path.join(tests_path, "model_devi.out")
        self.expect = np.array([0, 1.670048e-01, 4.182279e-04, 8.048649e-02, 5.095047e-01, 4.584241e-01, 4.819783e-01])
    
    def test_calc_model_devi(self):
        coord = np.load(os.path.join(self.data_dir, "set.000/coord.npy"))
        box = np.load(os.path.join(self.data_dir, "set.000/box.npy"))
        atype = np.loadtxt(os.path.join(self.data_dir, "type.raw"))
        model_devi = calc_model_devi(coord, box, atype, self.graphs)
        for ii in range(1, 7):
            self.assertAlmostEqual(model_devi[0][ii], self.expect[ii])
    
    def tearDown(self):
        for pb in self.graph_dirs:
            os.remove(pb)
        shutil.rmtree(self.data_dir)
