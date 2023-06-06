import json
import os
import unittest

import numpy as np
from common import (
    j_loader,
    run_dp,
    tests_path,
)
from packaging.version import parse as parse_version

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    tf,
)
from deepmd.infer import (
    DeepPot,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


def _file_delete(file):
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


def _init_models():
    data_file = str(tests_path / os.path.join("change_map", "data"))
    frozen_model = str(tests_path / "change_map_origin.pb")
    ckpt = str(tests_path / "change_map.ckpt")

    INPUT = str(tests_path / "input.json")
    jdata = j_loader(str(tests_path / os.path.join("change_map", "input.json")))

    jdata["training"]["save_ckpt"] = ckpt
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file

    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)
    ret = run_dp("dp train " + INPUT)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -c " + str(tests_path) + " -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")

    return INPUT, ckpt, frozen_model, data_file


if not parse_version(tf.__version__) < parse_version("1.15"):
    (
        INPUT,
        CKPT,
        FROZEN_MODEL,
        DATA,
    ) = _init_models()


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestDeepPotChangeMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.type_map = ["A", "O", "B", "H", "C"]
        cls.slim_type_map = ["O", "H"]
        cls.dp = DeepPot(FROZEN_MODEL)
        cls.slim_dp = DeepPot(cls.dp.change_map(cls.slim_type_map, data_sys=DATA))

    def setUp(self):
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
        self.atype = [1, 3, 3, 1, 3, 3]  # ["A", "O", "B", "H", "C"]
        self.slim_atype = [0, 1, 1, 0, 1, 1]  # ["O", "H"]
        self.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])

    @classmethod
    def tearDownClass(cls):
        _file_delete(INPUT)
        _file_delete(FROZEN_MODEL)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(CKPT + ".meta")
        _file_delete(CKPT + ".index")
        _file_delete(CKPT + ".data-00000-of-00001")
        _file_delete(CKPT + "-0.meta")
        _file_delete(CKPT + "-0.index")
        _file_delete(CKPT + "-0.data-00000-of-00001")
        _file_delete(CKPT + "-1.meta")
        _file_delete(CKPT + "-1.index")
        _file_delete(CKPT + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")
        cls.dp = None

    def test_equal(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic=False)
        ee_s, ff_s, vv_s = self.slim_dp.eval(
            self.coords, self.box, self.slim_atype, atomic=False
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ee_s.shape, (nframes, 1))
        self.assertEqual(ff_s.shape, (nframes, natoms, 3))
        self.assertEqual(vv_s.shape, (nframes, 9))
        # check values
        np.testing.assert_almost_equal(ff.ravel(), ff_s.ravel(), default_places)
        np.testing.assert_almost_equal(ee.ravel(), ee_s.ravel(), default_places)
        np.testing.assert_almost_equal(vv.ravel(), vv_s.ravel(), default_places)
