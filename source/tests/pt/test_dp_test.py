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
from deepmd.entrypoints.test import test_ener as dp_test_ener
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.data import (
    DeepmdData,
)

from .model.test_permutation import (
    model_property,
    model_se_e2_a,
    model_spin,
)


class DPTest:
    def test_dp_test_1_frame(self) -> None:
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
        dp_test(
            model=tmp_model.name,
            system=self.config["training"]["validation_data"]["systems"][0],
            datafile=None,
            set_prefix="set",
            numb_test=0,
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


class TestDPTestForceWeight(DPTest, unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_force_weight_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        system_dir = self._prepare_weighted_system()
        data_file = [system_dir]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.system_dir = system_dir
        self.input_json = "test_dp_test_force_weight.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def _prepare_weighted_system(self) -> str:
        src = Path(__file__).parent / "water/data/single"
        tmp_dir = tempfile.mkdtemp()
        shutil.copytree(src, tmp_dir, dirs_exist_ok=True)
        set_dir = Path(tmp_dir) / "set.000"
        forces = np.load(set_dir / "force.npy")
        forces[0, :3] += 1.0
        forces[0, -3:] += 10.0
        np.save(set_dir / "force.npy", forces)
        natoms = forces.shape[1] // 3
        atom_pref = np.ones((forces.shape[0], natoms), dtype=forces.dtype)
        atom_pref[:, 0] = 2.0
        atom_pref[:, -1] = 0.0
        np.save(set_dir / "atom_pref.npy", atom_pref)
        return tmp_dir

    def test_force_weight(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            trainer.get_data(is_train=False)
        model = torch.jit.script(trainer.model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)
        dp = DeepEval(tmp_model.name)
        data = DeepmdData(
            self.system_dir,
            set_prefix="set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )
        err = dp_test_ener(
            dp,
            data,
            self.system_dir,
            numb_test=1,
            detail_file=None,
            has_atom_ener=False,
        )
        test_data = data.get_test()
        coord = test_data["coord"].reshape([1, -1])
        box = test_data["box"][:1]
        atype = test_data["type"][0]
        ret = dp.eval(
            coord,
            box,
            atype,
            fparam=None,
            aparam=None,
            atomic=False,
            efield=None,
            mixed_type=False,
            spin=None,
        )
        force_pred = ret[1].reshape([1, -1])
        force_true = test_data["force"][:1]
        weight = test_data["atom_pref"][:1]
        diff = force_pred - force_true
        mae_unweighted = np.sum(np.abs(diff)) / diff.size
        rmse_unweighted = np.sqrt(np.sum(diff * diff) / diff.size)
        denom = weight.sum()
        mae_weighted = np.sum(np.abs(diff) * weight) / denom
        rmse_weighted = np.sqrt(np.sum(diff * diff * weight) / denom)
        np.testing.assert_allclose(err["mae_f"][0], mae_unweighted)
        np.testing.assert_allclose(err["rmse_f"][0], rmse_unweighted)
        np.testing.assert_allclose(err["mae_fw"][0], mae_weighted)
        np.testing.assert_allclose(err["rmse_fw"][0], rmse_weighted)
        os.unlink(tmp_model.name)

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.system_dir)


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
