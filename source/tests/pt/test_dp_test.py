# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.entrypoints.test import test as dp_test
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .model.test_permutation import (
    model_property,
    model_se_e2_a,
    model_spin,
)


class DPTest:
    def _run_dp_test(
        self, use_input_json: bool, numb_test: int = 0, use_train: bool = False
    ) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        has_spin = getattr(trainer.model, "has_spin", False)
        if callable(has_spin):
            has_spin = has_spin()
        if not has_spin:
            input_dict.pop("spin", None)
        input_dict["do_atomic_virial"] = True
        result = trainer.model(**input_dict)
        model = torch.jit.script(trainer.model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)
        val_sys = self.config["training"]["validation_data"]["systems"]
        if isinstance(val_sys, list):
            val_sys = val_sys[0]
        dp_test(
            model=tmp_model.name,
            system=val_sys,
            datafile=None,
            input_json=self.input_json if use_input_json else None,
            use_train=use_train,
            set_prefix="set",
            numb_test=numb_test,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=False,
        )
        os.unlink(tmp_model.name)
        natom = input_dict["atype"].shape[1]
        pred_e = np.loadtxt(self.detail_file + ".e.out", ndmin=2)[0, 1]
        np.testing.assert_almost_equal(
            pred_e,
            to_numpy_array(result["energy"])[0][0],
        )
        pred_e_peratom = np.loadtxt(self.detail_file + ".e_peratom.out", ndmin=2)[0, 1]
        np.testing.assert_almost_equal(pred_e_peratom, pred_e / natom)
        if not has_spin:
            pred_f = np.loadtxt(self.detail_file + ".f.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_f,
                to_numpy_array(result["force"]).reshape(-1, 3),
            )
            pred_v = np.loadtxt(self.detail_file + ".v.out", ndmin=2)[:, 9:18]
            np.testing.assert_almost_equal(
                pred_v,
                to_numpy_array(result["virial"]),
            )
            pred_v_peratom = np.loadtxt(self.detail_file + ".v_peratom.out", ndmin=2)[
                :, 9:18
            ]
            np.testing.assert_almost_equal(pred_v_peratom, pred_v / natom)
        else:
            pred_fr = np.loadtxt(self.detail_file + ".fr.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_fr,
                to_numpy_array(result["force"]).reshape(-1, 3),
            )
            pred_fm = np.loadtxt(self.detail_file + ".fm.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_fm,
                to_numpy_array(
                    result["force_mag"][result["mask_mag"].bool().squeeze(-1)]
                ).reshape(-1, 3),
            )

    def test_dp_test_1_frame(self) -> None:
        self._run_dp_test(False)

    def test_dp_test_input_json(self) -> None:
        self._run_dp_test(True)

    def test_dp_test_input_json_train(self) -> None:
        with open(self.input_json) as f:
            cfg = json.load(f)
        cfg["training"]["validation_data"]["systems"] = ["non-existent"]
        with open(self.input_json, "w") as f:
            json.dump(cfg, f, indent=4)
        self._run_dp_test(True, use_train=True)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestDPTestSeA(DPTest, unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)


class TestDPTestSeASpin(DPTest, unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_spin_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "NiO/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_spin)
        self.config["model"]["type_map"] = ["Ni", "O", "B"]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)


class TestDPTestSeARglob(unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_rglob_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        root_dir = str(Path(__file__).parent)
        self.config["training"]["validation_data"]["systems"] = root_dir
        self.config["training"]["validation_data"]["rglob_patterns"] = [
            "water/data/single"
        ]
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test_rglob.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def test_dp_test_input_json_rglob(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, _, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        model = torch.jit.script(trainer.model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)
        dp_test(
            model=tmp_model.name,
            system=self.config["training"]["validation_data"]["systems"],
            datafile=None,
            input_json=self.input_json,
            set_prefix="set",
            numb_test=1,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=False,
        )
        os.unlink(tmp_model.name)
        self.assertTrue(os.path.exists(self.detail_file + ".e.out"))

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestDPTestSeARglobTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_rglob_train_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        root_dir = str(Path(__file__).parent)
        self.config["training"]["training_data"]["systems"] = root_dir
        self.config["training"]["training_data"]["rglob_patterns"] = [
            "water/data/single"
        ]
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test_rglob_train.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def test_dp_test_input_json_rglob_train(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, _, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        model = torch.jit.script(trainer.model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)
        dp_test(
            model=tmp_model.name,
            system=self.config["training"]["validation_data"]["systems"],
            datafile=None,
            input_json=self.input_json,
            use_train=True,
            set_prefix="set",
            numb_test=1,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=False,
        )
        os.unlink(tmp_model.name)
        self.assertTrue(os.path.exists(self.detail_file + ".e.out"))

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestDPTestPropertySeA(unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_property_detail"
        input_json = str(Path(__file__).parent / "property/input.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "property/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_property)
        self.config["model"]["type_map"] = [
            self.config["model"]["type_map"][i] for i in [1, 0, 3, 2]
        ]
        self.input_json = "test_dp_test_property.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def test_dp_test_1_frame(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        result = trainer.model(**input_dict)
        model = torch.jit.script(trainer.model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)
        dp_test(
            model=tmp_model.name,
            system=self.config["training"]["validation_data"]["systems"][0],
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=True,
        )
        os.unlink(tmp_model.name)
        pred_property = np.loadtxt(self.detail_file + ".property.out.0")[:, 1]
        np.testing.assert_almost_equal(
            pred_property,
            to_numpy_array(result[model.get_var_name()])[0],
        )

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


if __name__ == "__main__":
    unittest.main()
