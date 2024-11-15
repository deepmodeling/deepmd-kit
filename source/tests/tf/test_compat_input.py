# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

from deepmd.tf.utils.compat import (
    convert_input_v0_v1,
    convert_input_v1_v2,
)

from .common import (
    j_loader,
)


class TestConvertInput(unittest.TestCase):
    def test_convert_smth(self) -> None:
        jdata0 = j_loader(os.path.join("compat_inputs", "water_se_a_v0.json"))
        jdata1 = j_loader(os.path.join("compat_inputs", "water_se_a_v1.json"))
        jdata = convert_input_v0_v1(jdata0, warning=False, dump=None)
        self.assertEqual(jdata, jdata1)

    def test_convert_nonsmth(self) -> None:
        jdata0 = j_loader(os.path.join("compat_inputs", "water_v0.json"))
        jdata1 = j_loader(os.path.join("compat_inputs", "water_v1.json"))
        jdata = convert_input_v0_v1(jdata0, warning=False, dump=None)
        self.assertEqual(jdata, jdata1)

    def test_convert_v1_v2(self) -> None:
        jdata0 = j_loader(os.path.join("compat_inputs", "water_v1.json"))
        jdata1 = j_loader(os.path.join("compat_inputs", "water_v2.json"))
        jdata = convert_input_v1_v2(jdata0, warning=False, dump=None)
        self.assertDictAlmostEqual(jdata, jdata1)

    def assertDictAlmostEqual(self, d1, d2, msg=None, places=7) -> None:
        self.assertEqual(d1.keys(), d2.keys())
        for kk, vv in d1.items():
            if isinstance(vv, dict):
                self.assertDictAlmostEqual(d1[kk], d2[kk], msg=msg)
            else:
                self.assertAlmostEqual(d1[kk], d2[kk], places=places, msg=msg)

    def test_json_yaml_equal(self) -> None:
        inputs = ("water_v1", "water_se_a_v1")

        for i in inputs:
            jdata = j_loader(os.path.join("yaml_inputs", f"{i}.json"))
            ydata = j_loader(os.path.join("yaml_inputs", f"{i}.yaml"))
            self.assertEqual(jdata, ydata)

        with self.assertRaises(TypeError):
            j_loader("path_with_wrong.extension")
