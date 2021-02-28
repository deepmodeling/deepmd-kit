import os,sys
import numpy as np
import unittest

from deepmd.utils.compat import convert_input_v0_v1
from common import j_loader

class TestConvertInput (unittest.TestCase) :
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

    def test_json_yaml_equal(self):

        inputs = ("water_v1", "water_se_a_v1")

        for i in inputs:
            jdata = j_loader(os.path.join('yaml_inputs', f'{i}.json'))
            ydata = j_loader(os.path.join('yaml_inputs', f'{i}.yaml'))
            self.assertEqual(jdata, ydata)

        with self.assertRaises(TypeError):
            j_loader("path_with_wrong.extension")

