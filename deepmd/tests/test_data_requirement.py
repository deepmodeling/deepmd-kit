import dpdata,os,sys,json,unittest
import numpy as np

from deepmd.common import data_requirement, add_data_requirement

class TestDataRequirement(unittest.TestCase):
    def test_add(self) :
        add_data_requirement('test', 3) 
        self.assertEqual(data_requirement['test']['ndof'], 3)
        self.assertEqual(data_requirement['test']['atomic'], False)
        self.assertEqual(data_requirement['test']['must'], False)
        self.assertEqual(data_requirement['test']['high_prec'], False)
        self.assertEqual(data_requirement['test']['repeat'], 1)
