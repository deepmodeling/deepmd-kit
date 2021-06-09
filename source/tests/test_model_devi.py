from operator import mod
import unittest
import os, glob
import numpy as np
from deepmd.entrypoints.model_devi import make_model_devi

class TestMakeModelDevi(unittest.TestCase):
    def setUp(self):
        self.wdir = os.path.join(os.path.dirname(__file__), "model_devi")
        self.data_dir = os.path.join(self.wdir, "data")
        self.graph_dirs = [os.path.join(self.wdir, f"graph.{ii:03}.pb") for ii in range(4)]
        self.output = os.path.join(self.wdir, "model_devi.out")
    
    def test_make_model_devi(self):
        make_model_devi(models=self.graph_dirs, system=self.data_dir, set_prefix="set", output=self.output, frequency=1, items='vf')
        self.assertTrue(os.path.exists(self.output))
        model_devi = np.loadtxt(self.output)
        self.assertEqual(model_devi.shape, (10, 7))
        self.assertAlmostEqual(model_devi[0, -3], 7.350327e-02)
    
    def tearDown(self):
        os.remove(self.output)
