# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import subprocess as sp
import unittest

import numpy as np

# from deepmd.entrypoints.compress import compress
from common import (
    j_loader,
    run_dp,
    tests_path,
)
from packaging.version import parse as parse_version

from deepmd.env import (
    tf,
)
from deepmd.infer import (
    DeepPot,
)


def _file_delete(file):
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


def _subprocess_run(command):
    popen = sp.Popen(command.split(), shell=False, stdout=sp.PIPE, stderr=sp.STDOUT)
    for line in iter(popen.stdout.readline, b""):
        if hasattr(line, "decode"):
            line = line.decode("utf-8")
        line = line.rstrip()
        print(line)
    popen.wait()
    return popen.returncode


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("2"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
def _init_models():
    data_file = str(tests_path / os.path.join("model_compression", "data"))
    inputs, frozen_models, compressed_models = [], [], []
    # 4 tests:
    # - type embedding FP64, se_atten FP64
    # - type embedding FP64, se_atten FP32
    # - type embedding FP32, se_atten FP64
    # - type embedding FP32, se_atten FP32
    tests = [
        {"se_atten precision": "float64", "type embedding precision": "float64"},
        {"se_atten precision": "float64", "type embedding precision": "float32"},
        {"se_atten precision": "float32", "type embedding precision": "float64"},
        {"se_atten precision": "float32", "type embedding precision": "float32"},
    ]
    for i in range(4):
        INPUT = str(tests_path / f"input{i}.json")
        frozen_model = str(tests_path / f"dp-original-se-atten{i}.pb")
        compressed_model = str(tests_path / f"dp-compressed-se-atten{i}.pb")
        jdata = j_loader(
            str(tests_path / os.path.join("model_compression", "input.json"))
        )
        jdata["model"]["descriptor"] = {}
        jdata["model"]["descriptor"]["type"] = "se_atten"
        jdata["model"]["descriptor"]["precision"] = tests[i]["se_atten precision"]
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["sel"] = 120
        jdata["model"]["descriptor"]["attn_layer"] = 0
        jdata["model"]["type_embedding"] = {}
        jdata["model"]["type_embedding"]["precision"] = tests[i][
            "type embedding precision"
        ]
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

        inputs.append(INPUT)
        frozen_models.append(frozen_model)
        compressed_models.append(compressed_model)

    return inputs, frozen_models, compressed_models


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("2"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
def _init_models_exclude_types():
    data_file = str(tests_path / os.path.join("model_compression", "data"))
    inputs, frozen_models, compressed_models = [], [], []
    # 4 tests:
    # - type embedding FP64, se_atten FP64
    # - type embedding FP64, se_atten FP32
    # - type embedding FP32, se_atten FP64
    # - type embedding FP32, se_atten FP32
    tests = [
        {"se_atten precision": "float64", "type embedding precision": "float64"},
        {"se_atten precision": "float64", "type embedding precision": "float32"},
        {"se_atten precision": "float32", "type embedding precision": "float64"},
        {"se_atten precision": "float32", "type embedding precision": "float32"},
    ]
    for i in range(4):
        INPUT = str(tests_path / f"input{i}.json")
        frozen_model = str(tests_path / f"dp-original-se-atten{i}-exclude-types.pb")
        compressed_model = str(
            tests_path / f"dp-compressed-se-atten{i}-exclude-types.pb"
        )
        jdata = j_loader(
            str(tests_path / os.path.join("model_compression", "input.json"))
        )
        jdata["model"]["descriptor"] = {}
        jdata["model"]["descriptor"]["type"] = "se_atten"
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 1]]
        jdata["model"]["descriptor"]["precision"] = tests[i]["se_atten precision"]
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        jdata["model"]["descriptor"]["sel"] = 120
        jdata["model"]["descriptor"]["attn_layer"] = 0
        jdata["model"]["type_embedding"] = {}
        jdata["model"]["type_embedding"]["precision"] = tests[i][
            "type embedding precision"
        ]
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

        inputs.append(INPUT)
        frozen_models.append(frozen_model)
        compressed_models.append(compressed_model)

    return inputs, frozen_models, compressed_models


INPUTS, FROZEN_MODELS, COMPRESSED_MODELS = _init_models()
INPUTS_ET, FROZEN_MODELS_ET, COMPRESSED_MODELS_ET = _init_models_exclude_types()


def _get_default_places(nth_test):
    return 10 if nth_test == 0 else 3


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("2"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestDeepPotAPBC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dp_originals = [DeepPot(FROZEN_MODELS[i]) for i in range(4)]
        self.dp_compresseds = [DeepPot(COMPRESSED_MODELS[i]) for i in range(4)]
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

    def test_attrs(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            self.assertEqual(dp_original.get_ntypes(), 2)
            self.assertAlmostEqual(dp_original.get_rcut(), 6.0, places=default_places)
            self.assertEqual(dp_original.get_type_map(), ["O", "H"])
            self.assertEqual(dp_original.get_dim_fparam(), 0)
            self.assertEqual(dp_original.get_dim_aparam(), 0)

            self.assertEqual(dp_compressed.get_ntypes(), 2)
            self.assertAlmostEqual(dp_compressed.get_rcut(), 6.0, places=default_places)
            self.assertEqual(dp_compressed.get_type_map(), ["O", "H"])
            self.assertEqual(dp_compressed.get_dim_fparam(), 0)
            self.assertEqual(dp_compressed.get_dim_aparam(), 0)

    def test_1frame(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=False
            )
            ee1, ff1, vv1 = dp_compressed.eval(
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

    def test_1frame_atm(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0, ae0, av0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=True
            )
            ee1, ff1, vv1, ae1, av1 = dp_compressed.eval(
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

    def test_2frame_atm(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            coords2 = np.concatenate((self.coords, self.coords))
            box2 = np.concatenate((self.box, self.box))
            ee0, ff0, vv0, ae0, av0 = dp_original.eval(
                coords2, box2, self.atype, atomic=True
            )
            ee1, ff1, vv1, ae1, av1 = dp_compressed.eval(
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


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("2"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestDeepPotANoPBC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dp_originals = [DeepPot(FROZEN_MODELS[i]) for i in range(4)]
        self.dp_compresseds = [DeepPot(COMPRESSED_MODELS[i]) for i in range(4)]
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
        self.box = None

    def test_1frame(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=False
            )
            ee1, ff1, vv1 = dp_compressed.eval(
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

    def test_1frame_atm(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0, ae0, av0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=True
            )
            ee1, ff1, vv1, ae1, av1 = dp_compressed.eval(
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

    def test_2frame_atm(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            coords2 = np.concatenate((self.coords, self.coords))
            ee0, ff0, vv0, ae0, av0 = dp_original.eval(
                coords2, self.box, self.atype, atomic=True
            )
            ee1, ff1, vv1, ae1, av1 = dp_compressed.eval(
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


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("2"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestDeepPotALargeBoxNoPBC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dp_originals = [DeepPot(FROZEN_MODELS[i]) for i in range(4)]
        self.dp_compresseds = [DeepPot(COMPRESSED_MODELS[i]) for i in range(4)]
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
        self.box = np.array([19.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])

    def test_1frame(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=False
            )
            ee1, ff1, vv1 = dp_compressed.eval(
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

    def test_1frame_atm(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0, ae0, av0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=True
            )
            ee1, ff1, vv1, ae1, av1 = dp_compressed.eval(
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

    def test_ase(self):
        for i in range(4):
            default_places = _get_default_places(i)
            from ase import (
                Atoms,
            )

            from deepmd.calculator import (
                DP,
            )

            water0 = Atoms(
                "OHHOHH",
                positions=self.coords.reshape((-1, 3)),
                cell=self.box.reshape((3, 3)),
                calculator=DP(FROZEN_MODELS[i]),
            )
            water1 = Atoms(
                "OHHOHH",
                positions=self.coords.reshape((-1, 3)),
                cell=self.box.reshape((3, 3)),
                calculator=DP(COMPRESSED_MODELS[i]),
            )
            ee0 = water0.get_potential_energy()
            ff0 = water0.get_forces()
            ee1 = water1.get_potential_energy()
            ff1 = water1.get_forces()
            nframes = 1
            np.testing.assert_almost_equal(ff0, ff1, default_places)
            np.testing.assert_almost_equal(ee0, ee1, default_places)


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("2"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestDeepPotAPBCExcludeTypes(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dp_originals = [DeepPot(FROZEN_MODELS_ET[i]) for i in range(4)]
        self.dp_compresseds = [DeepPot(COMPRESSED_MODELS_ET[i]) for i in range(4)]
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

    @classmethod
    def tearDownClass(self):
        for i in range(4):
            _file_delete(INPUTS_ET[i])
            _file_delete(FROZEN_MODELS_ET[i])
            _file_delete(COMPRESSED_MODELS_ET[i])
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

    def test_attrs(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            self.assertEqual(dp_original.get_ntypes(), 2)
            self.assertAlmostEqual(dp_original.get_rcut(), 6.0, places=default_places)
            self.assertEqual(dp_original.get_type_map(), ["O", "H"])
            self.assertEqual(dp_original.get_dim_fparam(), 0)
            self.assertEqual(dp_original.get_dim_aparam(), 0)

            self.assertEqual(dp_compressed.get_ntypes(), 2)
            self.assertAlmostEqual(dp_compressed.get_rcut(), 6.0, places=default_places)
            self.assertEqual(dp_compressed.get_type_map(), ["O", "H"])
            self.assertEqual(dp_compressed.get_dim_fparam(), 0)
            self.assertEqual(dp_compressed.get_dim_aparam(), 0)

    def test_1frame(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=False
            )
            ee1, ff1, vv1 = dp_compressed.eval(
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

    def test_1frame_atm(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            ee0, ff0, vv0, ae0, av0 = dp_original.eval(
                self.coords, self.box, self.atype, atomic=True
            )
            ee1, ff1, vv1, ae1, av1 = dp_compressed.eval(
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

    def test_2frame_atm(self):
        for i in range(4):
            dp_original = self.dp_originals[i]
            dp_compressed = self.dp_compresseds[i]
            default_places = _get_default_places(i)

            coords2 = np.concatenate((self.coords, self.coords))
            box2 = np.concatenate((self.box, self.box))
            ee0, ff0, vv0, ae0, av0 = dp_original.eval(
                coords2, box2, self.atype, atomic=True
            )
            ee1, ff1, vv1, ae1, av1 = dp_compressed.eval(
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
