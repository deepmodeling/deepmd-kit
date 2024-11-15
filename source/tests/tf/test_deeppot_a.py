# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    MODEL_VERSION,
)
from deepmd.tf.infer import (
    DeepPot,
)
from deepmd.tf.utils.convert import (
    convert_dp10_to_dp11,
    convert_dp012_to_dp10,
    convert_dp12_to_dp13,
    convert_dp13_to_dp20,
    convert_dp20_to_dp21,
    convert_pbtxt_to_pb,
    detect_model_version,
)

from .common import (
    infer_path,
    run_dp,
    tests_path,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


class TestModelMajorCompatability(unittest.TestCase):
    def setUp(self) -> None:
        model_file = str(infer_path / os.path.join("deeppot.pbtxt"))
        with open(model_file) as fp:
            # data = fp.read().replace('\n', '')
            data = fp.read().split("\n")
            for ii in range(len(data)):
                if "model_attr/model_version" in data[ii]:
                    for jj in range(ii, len(data)):
                        if "string_val:" in data[jj]:
                            data[jj] = data[jj].replace(MODEL_VERSION, "0.0")
                            break
        self.version_pbtxt = str(tests_path / "deeppot-ver.pbtxt")
        self.version_pb = str(tests_path / "deeppot.pb")
        with open(self.version_pbtxt, "w") as fp:
            fp.write("\n".join(data))
        convert_pbtxt_to_pb(self.version_pbtxt, self.version_pb)

    def tearDown(self) -> None:
        os.remove(self.version_pbtxt)
        os.remove(self.version_pb)

    def test(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            DeepPot(str(self.version_pb))
        self.assertTrue("incompatible" in str(context.exception))
        self.assertTrue(MODEL_VERSION in str(context.exception))
        self.assertTrue("0.0" in str(context.exception))


class TestModelMinorCompatability(unittest.TestCase):
    def setUp(self) -> None:
        model_file = str(infer_path / os.path.join("deeppot.pbtxt"))
        with open(model_file) as fp:
            # data = fp.read().replace('\n', '')
            data = fp.read().split("\n")
            for ii in range(len(data)):
                if "model_attr/model_version" in data[ii]:
                    for jj in range(ii, len(data)):
                        if "string_val:" in data[jj]:
                            data[jj] = data[jj].replace(MODEL_VERSION, "0.1000000")
                            break
        self.version_pbtxt = str(tests_path / "deeppot-ver.pbtxt")
        self.version_pb = str(tests_path / "deeppot.pb")
        with open(self.version_pbtxt, "w") as fp:
            fp.write("\n".join(data))
        convert_pbtxt_to_pb(self.version_pbtxt, self.version_pb)

    def tearDown(self) -> None:
        os.remove(self.version_pbtxt)
        os.remove(self.version_pb)

    def test(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            DeepPot(self.version_pb)
        self.assertTrue("incompatible" in str(context.exception))
        self.assertTrue(MODEL_VERSION in str(context.exception))
        self.assertTrue("0.1000000" in str(context.exception))


# TestDeepPotAPBC, TestDeepPotANoPBC, TestDeepPotALargeBoxNoPBC: moved to infer/test_models.py


class TestModelConvert(unittest.TestCase):
    def setUp(self) -> None:
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ]
        )
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])

    def test_convert_012(self) -> None:
        old_model = "deeppot.pb"
        new_model = "deeppot-new.pb"
        convert_pbtxt_to_pb(str(infer_path / "sea_012.pbtxt"), old_model)
        run_dp(f"dp convert-from 0.12 -i {old_model} -o {new_model}")
        dp = DeepPot(new_model)
        _ = dp.eval(self.coords, self.box, self.atype, atomic=True)
        os.remove(old_model)
        os.remove(new_model)

    def test_convert(self) -> None:
        old_model = "deeppot.pb"
        new_model = "deeppot-new.pb"
        convert_pbtxt_to_pb(str(infer_path / "sea_012.pbtxt"), old_model)
        run_dp(f"dp convert-from -i {old_model} -o {new_model}")
        dp = DeepPot(new_model)
        _ = dp.eval(self.coords, self.box, self.atype, atomic=True)
        os.remove(old_model)
        os.remove(new_model)

    def test_detect(self) -> None:
        old_model = "deeppot.pb"
        new_model_txt = "deeppot_new.pbtxt"
        new_model_pb = "deeppot_new.pb"
        convert_pbtxt_to_pb(str(infer_path / "sea_012.pbtxt"), old_model)
        version = detect_model_version(old_model)
        self.assertEqual(version, parse_version("0.12"))
        os.remove(old_model)
        shutil.copyfile(str(infer_path / "sea_012.pbtxt"), new_model_txt)
        convert_dp012_to_dp10(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("1.0"))
        os.remove(new_model_pb)
        convert_dp10_to_dp11(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("1.3"))
        os.remove(new_model_pb)
        convert_dp12_to_dp13(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("1.3"))
        os.remove(new_model_pb)
        convert_dp13_to_dp20(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("2.0"))
        os.remove(new_model_pb)
        convert_dp20_to_dp21(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("2.1"))
        os.remove(new_model_pb)
        os.remove(new_model_txt)


class TestTypeEmbed(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("se_e2_a_tebd.pbtxt")),
            "se_e2_a_tebd.pb",
        )
        cls.dp = DeepPot("se_e2_a_tebd.pb")

    def test_eval_typeebd(self) -> None:
        expected_typeebd = np.array(
            [
                [
                    -0.4602908199,
                    -0.9440795817,
                    -0.857044451,
                    -0.3448434537,
                    -0.6310194663,
                    -0.9765837147,
                    -0.3945653821,
                    0.8973716518,
                ],
                [
                    -0.7239568558,
                    -0.9672733137,
                    -0.420987752,
                    -0.4542931277,
                    -0.79586188,
                    -0.9615886543,
                    -0.6864800369,
                    0.9477863254,
                ],
            ]
        )

        eval_typeebd = self.dp.eval_typeebd()
        np.testing.assert_almost_equal(eval_typeebd, expected_typeebd, default_places)


# TestFparamAparam: moved to infer/test_models.py
# TestDeepPotAPBCNeighborList: moved to infer/test_models.py
