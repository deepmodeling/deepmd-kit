import os,sys
import numpy as np
import unittest

from deepmd.compat import convert_input_v0_v1
from deepmd.common import j_loader

class TestConvertInputV0V1 (unittest.TestCase) :
    def test_convert_smth(self):
        jdata0 = j_loader(os.path.join('compat_inputs', 'water_se_a_v0.json'))
        jdata1 = j_loader(os.path.join('compat_inputs', 'water_se_a_v1.json'))
        jdata = convert_input_v0_v1(jdata0, warning = False, dump = None)
        self.assertEqual(jdata, jdata1)

    def test_convert_nonsmth(self):
        jdata0 = j_loader(os.path.join('compat_inputs', 'water_v0.json'))
        jdata1 = j_loader(os.path.join('compat_inputs', 'water_v1.json'))
        jdata = convert_input_v0_v1(jdata0, warning = False, dump = None)
        self.assertEqual(jdata, jdata1)

