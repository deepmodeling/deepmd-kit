import os,sys,json
import numpy as np
import unittest

from deepmd.compat import convert_input_v0_v1

class TestConvertInputV0V1 (unittest.TestCase) :
    def test_convert_smth(self):
        with open(os.path.join('compat_inputs', 'water_se_a_v0.json')) as fp:
            jdata0 = json.load(fp)
        with open(os.path.join('compat_inputs', 'water_se_a_v1.json')) as fp:
            jdata1 = json.load(fp)
        jdata = convert_input_v0_v1(jdata0, warning = False, dump = None)
        self.assertEqual(jdata, jdata1)

    def test_convert_nonsmth(self):
        with open(os.path.join('compat_inputs', 'water_v0.json')) as fp:
            jdata0 = json.load(fp)
        with open(os.path.join('compat_inputs', 'water_v1.json')) as fp:
            jdata1 = json.load(fp)
        jdata = convert_input_v0_v1(jdata0, warning = False, dump = None)
        self.assertEqual(jdata, jdata1)

