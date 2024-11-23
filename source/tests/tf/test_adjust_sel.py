# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.tf.infer import (
    DeepPot,
)

# from deepmd.tf.entrypoints.compress import compress
from .common import (
    j_loader,
    run_dp,
    tests_path,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


def _file_delete(file) -> None:
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


def _init_models():
    # we use the setting for model compression
    data_file = str(tests_path / os.path.join("model_compression", "data"))
    frozen_model = str(tests_path / "dp-adjust-sel-original.pb")
    decreased_model = str(tests_path / "dp-adjust-sel-original-decreased.pb")
    increased_model = str(tests_path / "dp-adjust-sel-original-increased.pb")
    INPUT = str(tests_path / "input.json")
    jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file
    jdata["model"]["descriptor"]["set_davg_zero"] = True
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)

    ret = run_dp("dp train " + INPUT + " --skip-neighbor-stat")
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")

    jdata["training"]["numb_steps"] = 0
    jdata["model"]["descriptor"]["sel"] = [2, 4]  # equal to data
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)
    ret = run_dp("dp train " + INPUT + " -f " + frozen_model + " --skip-neighbor-stat")
    np.testing.assert_equal(ret, 0, "DP model adjust sel failed!")
    ret = run_dp("dp freeze -o " + decreased_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")

    jdata["model"]["descriptor"]["sel"] = [300, 300]  # equal to data
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)
    ret = run_dp("dp train " + INPUT + " -f " + frozen_model + " --skip-neighbor-stat")
    np.testing.assert_equal(ret, 0, "DP model adjust sel failed!")
    ret = run_dp("dp freeze -o " + increased_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")
    return INPUT, frozen_model, decreased_model, increased_model


class TestDeepPotAAdjustSel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        INPUT, FROZEN_MODEL, DECREASED_MODEL, INCREASED_MODEL = _init_models()
        cls.dp_original = DeepPot(FROZEN_MODEL)
        cls.dp_decreased = DeepPot(DECREASED_MODEL)
        cls.dp_increased = DeepPot(INCREASED_MODEL)
        cls.coords = np.array(
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
        cls.atype = [0, 1, 1, 0, 1, 1]
        cls.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])

    def test_attrs(self) -> None:
        self.assertEqual(self.dp_original.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_original.get_rcut(), 6.0, places=default_places)
        self.assertEqual(self.dp_original.get_type_map(), ["O", "H"])
        self.assertEqual(self.dp_original.get_dim_fparam(), 0)
        self.assertEqual(self.dp_original.get_dim_aparam(), 0)

        self.assertEqual(self.dp_decreased.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_decreased.get_rcut(), 6.0, places=default_places)
        self.assertEqual(self.dp_decreased.get_type_map(), ["O", "H"])
        self.assertEqual(self.dp_decreased.get_dim_fparam(), 0)
        self.assertEqual(self.dp_decreased.get_dim_aparam(), 0)

        self.assertEqual(self.dp_increased.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_increased.get_rcut(), 6.0, places=default_places)
        self.assertEqual(self.dp_increased.get_type_map(), ["O", "H"])
        self.assertEqual(self.dp_increased.get_dim_fparam(), 0)
        self.assertEqual(self.dp_increased.get_dim_aparam(), 0)

    def test_1frame(self) -> None:
        ee0, ff0, vv0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=False
        )
        ee1, ff1, vv1 = self.dp_decreased.eval(
            self.coords, self.box, self.atype, atomic=False
        )
        ee2, ff2, vv2 = self.dp_increased.eval(
            self.coords, self.box, self.atype, atomic=False
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes, 1))
        self.assertEqual(ff0.shape, (nframes, natoms, 3))
        self.assertEqual(vv0.shape, (nframes, 9))
        self.assertEqual(ee1.shape, (nframes, 1))
        self.assertEqual(ff1.shape, (nframes, natoms, 3))
        self.assertEqual(vv1.shape, (nframes, 9))
        self.assertEqual(ee2.shape, (nframes, 1))
        self.assertEqual(ff2.shape, (nframes, natoms, 3))
        self.assertEqual(vv2.shape, (nframes, 9))
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)
        np.testing.assert_almost_equal(ff0, ff2, default_places)
        np.testing.assert_almost_equal(ee0, ee2, default_places)
        np.testing.assert_almost_equal(vv0, vv2, default_places)

    def test_1frame_atm(self) -> None:
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_decreased.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        ee2, ff2, vv2, ae2, av2 = self.dp_increased.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes, 1))
        self.assertEqual(ff0.shape, (nframes, natoms, 3))
        self.assertEqual(vv0.shape, (nframes, 9))
        self.assertEqual(ae0.shape, (nframes, natoms, 1))
        self.assertEqual(av0.shape, (nframes, natoms, 9))
        self.assertEqual(ee1.shape, (nframes, 1))
        self.assertEqual(ff1.shape, (nframes, natoms, 3))
        self.assertEqual(vv1.shape, (nframes, 9))
        self.assertEqual(ae1.shape, (nframes, natoms, 1))
        self.assertEqual(av1.shape, (nframes, natoms, 9))
        self.assertEqual(ee2.shape, (nframes, 1))
        self.assertEqual(ff2.shape, (nframes, natoms, 3))
        self.assertEqual(vv2.shape, (nframes, 9))
        self.assertEqual(ae2.shape, (nframes, natoms, 1))
        self.assertEqual(av2.shape, (nframes, natoms, 9))
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)
        np.testing.assert_almost_equal(ff0, ff2, default_places)
        np.testing.assert_almost_equal(ae0, ae2, default_places)
        np.testing.assert_almost_equal(av0, av2, default_places)
        np.testing.assert_almost_equal(ee0, ee2, default_places)
        np.testing.assert_almost_equal(vv0, vv2, default_places)

    def test_2frame_atm(self) -> None:
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            coords2, box2, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_decreased.eval(
            coords2, box2, self.atype, atomic=True
        )
        ee2, ff2, vv2, ae2, av2 = self.dp_increased.eval(
            coords2, box2, self.atype, atomic=True
        )
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes, 1))
        self.assertEqual(ff0.shape, (nframes, natoms, 3))
        self.assertEqual(vv0.shape, (nframes, 9))
        self.assertEqual(ae0.shape, (nframes, natoms, 1))
        self.assertEqual(av0.shape, (nframes, natoms, 9))
        self.assertEqual(ee1.shape, (nframes, 1))
        self.assertEqual(ff1.shape, (nframes, natoms, 3))
        self.assertEqual(vv1.shape, (nframes, 9))
        self.assertEqual(ae1.shape, (nframes, natoms, 1))
        self.assertEqual(av1.shape, (nframes, natoms, 9))
        self.assertEqual(ee2.shape, (nframes, 1))
        self.assertEqual(ff2.shape, (nframes, natoms, 3))
        self.assertEqual(vv2.shape, (nframes, 9))
        self.assertEqual(ae2.shape, (nframes, natoms, 1))
        self.assertEqual(av2.shape, (nframes, natoms, 9))

        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)
        np.testing.assert_almost_equal(ff0, ff2, default_places)
        np.testing.assert_almost_equal(ae0, ae2, default_places)
        np.testing.assert_almost_equal(av0, av2, default_places)
        np.testing.assert_almost_equal(ee0, ee2, default_places)
        np.testing.assert_almost_equal(vv0, vv2, default_places)

    def test_descriptor(self) -> None:
        dd0 = self.dp_original.eval_descriptor(self.coords, self.box, self.atype)
        dd1 = self.dp_decreased.eval_descriptor(self.coords, self.box, self.atype)
        dd2 = self.dp_increased.eval_descriptor(self.coords, self.box, self.atype)
        # check shape of the returns
        self.assertEqual(dd0.shape, dd1.shape)
        self.assertEqual(dd0.shape, dd2.shape)
        # check values
        np.testing.assert_almost_equal(dd0, dd1, default_places)
        np.testing.assert_almost_equal(dd0, dd2, default_places)
