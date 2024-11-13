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
    data_file = str(tests_path / os.path.join("model_compression", "data"))
    frozen_model = str(tests_path / "dp-original.pb")
    compressed_model = str(tests_path / "dp-compressed.pb")
    INPUT = str(tests_path / "input.json")
    jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)

    ret = run_dp("dp train " + INPUT)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")
    ret = run_dp("dp compress " + " -i " + frozen_model + " -o " + compressed_model)
    np.testing.assert_equal(ret, 0, "DP model compression failed!")
    return INPUT, frozen_model, compressed_model


def _init_models_exclude_types():
    data_file = str(tests_path / os.path.join("model_compression", "data"))
    frozen_model = str(tests_path / "dp-original-exclude-types.pb")
    compressed_model = str(tests_path / "dp-compressed-exclude-types.pb")
    INPUT = str(tests_path / "input.json")
    jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
    jdata["model"]["descriptor"]["exclude_types"] = [[0, 1]]
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)

    ret = run_dp("dp train " + INPUT)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")
    ret = run_dp("dp compress " + " -i " + frozen_model + " -o " + compressed_model)
    np.testing.assert_equal(ret, 0, "DP model compression failed!")
    return INPUT, frozen_model, compressed_model


def setUpModule() -> None:
    global \
        INPUT, \
        FROZEN_MODEL, \
        COMPRESSED_MODEL, \
        INPUT_ET, \
        FROZEN_MODEL_ET, \
        COMPRESSED_MODEL_ET
    INPUT, FROZEN_MODEL, COMPRESSED_MODEL = _init_models()
    INPUT_ET, FROZEN_MODEL_ET, COMPRESSED_MODEL_ET = _init_models_exclude_types()


class TestDeepPotAPBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dp_original = DeepPot(FROZEN_MODEL)
        cls.dp_compressed = DeepPot(COMPRESSED_MODEL)
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

        self.assertEqual(self.dp_compressed.get_ntypes(), 2)
        self.assertAlmostEqual(
            self.dp_compressed.get_rcut(), 6.0, places=default_places
        )
        self.assertEqual(self.dp_compressed.get_type_map(), ["O", "H"])
        self.assertEqual(self.dp_compressed.get_dim_fparam(), 0)
        self.assertEqual(self.dp_compressed.get_dim_aparam(), 0)

    def test_1frame(self) -> None:
        ee0, ff0, vv0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=False
        )
        ee1, ff1, vv1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_1frame_atm(self) -> None:
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_2frame_atm(self) -> None:
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            coords2, box2, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
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

        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)


class TestDeepPotANoPBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dp_original = DeepPot(FROZEN_MODEL)
        cls.dp_compressed = DeepPot(COMPRESSED_MODEL)
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
        cls.box = None

    def test_1frame(self) -> None:
        ee0, ff0, vv0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=False
        )
        ee1, ff1, vv1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_1frame_atm(self) -> None:
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_2frame_atm(self) -> None:
        coords2 = np.concatenate((self.coords, self.coords))
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            coords2, self.box, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
            coords2, self.box, self.atype, atomic=True
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

        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)


class TestDeepPotALargeBoxNoPBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dp_original = DeepPot(FROZEN_MODEL)
        cls.dp_compressed = DeepPot(COMPRESSED_MODEL)
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
        cls.box = np.array([19.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])

    def test_1frame(self) -> None:
        ee0, ff0, vv0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=False
        )
        ee1, ff1, vv1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_1frame_atm(self) -> None:
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_ase(self) -> None:
        from ase import (
            Atoms,
        )

        from deepmd.tf.calculator import (
            DP,
        )

        water0 = Atoms(
            "OHHOHH",
            positions=self.coords.reshape((-1, 3)),
            cell=self.box.reshape((3, 3)),
            calculator=DP(FROZEN_MODEL),
        )
        water1 = Atoms(
            "OHHOHH",
            positions=self.coords.reshape((-1, 3)),
            cell=self.box.reshape((3, 3)),
            calculator=DP(COMPRESSED_MODEL),
        )
        ee0 = water0.get_potential_energy()
        ff0 = water0.get_forces()
        ee1 = water1.get_potential_energy()
        ff1 = water1.get_forces()
        nframes = 1
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)


class TestDeepPotAPBCExcludeTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dp_original = DeepPot(FROZEN_MODEL_ET)
        cls.dp_compressed = DeepPot(COMPRESSED_MODEL_ET)
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

    @classmethod
    def tearDownClass(cls) -> None:
        _file_delete(INPUT_ET)
        _file_delete(FROZEN_MODEL_ET)
        _file_delete(COMPRESSED_MODEL_ET)
        _file_delete("out.json")
        _file_delete("compress.json")
        _file_delete("checkpoint")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")
        _file_delete("model.ckpt-100.meta")
        _file_delete("model.ckpt-100.index")
        _file_delete("model.ckpt-100.data-00000-of-00001")
        _file_delete("model-compression/checkpoint")
        _file_delete("model-compression/model.ckpt.meta")
        _file_delete("model-compression/model.ckpt.index")
        _file_delete("model-compression/model.ckpt.data-00000-of-00001")
        _file_delete("model-compression")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_attrs(self) -> None:
        self.assertEqual(self.dp_original.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp_original.get_rcut(), 6.0, places=default_places)
        self.assertEqual(self.dp_original.get_type_map(), ["O", "H"])
        self.assertEqual(self.dp_original.get_dim_fparam(), 0)
        self.assertEqual(self.dp_original.get_dim_aparam(), 0)

        self.assertEqual(self.dp_compressed.get_ntypes(), 2)
        self.assertAlmostEqual(
            self.dp_compressed.get_rcut(), 6.0, places=default_places
        )
        self.assertEqual(self.dp_compressed.get_type_map(), ["O", "H"])
        self.assertEqual(self.dp_compressed.get_dim_fparam(), 0)
        self.assertEqual(self.dp_compressed.get_dim_aparam(), 0)

    def test_1frame(self) -> None:
        ee0, ff0, vv0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=False
        )
        ee1, ff1, vv1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_1frame_atm(self) -> None:
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
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
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_2frame_atm(self) -> None:
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            coords2, box2, self.atype, atomic=True
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
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

        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)
