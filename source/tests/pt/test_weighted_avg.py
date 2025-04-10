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

from deepmd.entrypoints.test import test_ener as dp_test_ener
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.weight_avg import (
    weighted_average,
)

from .model.test_permutation import (
    model_se_e2_a,
    model_spin,
)


class Test_testener_without_spin(unittest.TestCase):
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
        trainer = get_trainer(deepcopy(self.config))
        model = torch.jit.script(trainer.model)
        self.tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, self.tmp_model.name)

    def test_dp_test_ener_without_spin(self) -> None:
        dp = DeepEval(self.tmp_model.name, head="PyTorch")
        system = self.config["training"]["validation_data"]["systems"][0]
        data = DeepmdData(
            sys_path=system,
            set_prefix="set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )
        err = dp_test_ener(
            dp,
            data,
            system,
            numb_test=1,
            detail_file=None,
            has_atom_ener=False,
        )
        self.assertIn("mae_e", err, "'mae_e' key is missing in the result")
        self.assertNotIn(
            "mae_fm", err, "'mae_fm' key should not be present in the result"
        )
        self.assertNotIn(
            "mae_v", err, "'mae_v' key should not be present in the result"
        )
        self.assertIn("mae_f", err, "'mae_f' key is missing in the result")

    def test_dp_test_ener_with_multisys_and_with_virial(self) -> None:
        dp = DeepEval(self.tmp_model.name, head="PyTorch")
        system = self.config["training"]["validation_data"]["systems"][0]
        data = DeepmdData(
            sys_path=system,
            set_prefix="set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )
        err = []
        err_novirial = dp_test_ener(
            dp,
            data,
            system,
            numb_test=1,
            detail_file=None,
            has_atom_ener=False,
        )
        err.append(err_novirial)
        ener_nv, weight_nv = err_novirial["mae_e"]
        virial_path_fake = os.path.join(
            self.config["training"]["validation_data"]["systems"][0],
            "set.000",
            "virial.npy",
        )
        np.save(virial_path_fake, np.ones([1, 9], dtype=np.float64))
        data = DeepmdData(
            sys_path=system,
            set_prefix="set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )
        err_virial = dp_test_ener(
            dp,
            data,
            system,
            numb_test=1,
            detail_file=None,
            has_atom_ener=False,
        )

        self.assertIn("mae_e", err_virial, "'mae_e' key is missing in the result")
        self.assertNotIn(
            "mae_fm", err_virial, "'mae_fm' key should not be present in the result"
        )
        self.assertIn("mae_v", err_virial, "'mae_v' key is missing in the result")
        self.assertIn("mae_f", err_virial, "'mae_f' key is missing in the result")

        ener_v, weight_v = err_virial["mae_e"]
        mae_v, _ = err_virial["mae_v"]
        weight = weight_nv + weight_v
        ener = (ener_v * weight_v) + (ener_nv * weight_nv)
        mae_e_expected = ener / weight
        err.append(err_virial)
        avg_err = weighted_average(err)

        self.assertEqual(
            avg_err["mae_v"],
            mae_v,
            f"Expected mae_v in avg_err to be {mae_v} but got {avg_err['mae_v']}",
        )

        self.assertEqual(
            avg_err["mae_e"],
            mae_e_expected,
            f"Expected mae_e in avg_err to be {mae_e_expected} but got {avg_err['mae_e']}",
        )

        os.unlink(self.tmp_model.name)

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
            virial_path_fake = os.path.join(
                self.config["training"]["validation_data"]["systems"][0],
                "set.000",
                "virial.npy",
            )
            if os.path.exists(virial_path_fake):
                os.remove(virial_path_fake)


class Test_testener_spin(unittest.TestCase):
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
        trainer = get_trainer(deepcopy(self.config))
        model = torch.jit.script(trainer.model)
        self.tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, self.tmp_model.name)

    def test_dp_test_ener_with_spin(self) -> None:
        dp = DeepEval(self.tmp_model.name, head="PyTorch")
        system = self.config["training"]["validation_data"]["systems"][0]
        data = DeepmdData(
            sys_path=system,
            set_prefix="set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )

        err = dp_test_ener(
            dp,
            data,
            system,
            numb_test=1,
            detail_file=None,
            has_atom_ener=False,
        )
        self.assertIn("mae_e", err, "'mae_e' key is missing in the result")
        self.assertIn("mae_fm", err, "'mae_fm' key is missing in the result")
        self.assertNotIn(
            "mae_v", err, "'mae_v' key should not be present in the result"
        )
        self.assertNotIn(
            "mae_f", err, "'mae_f' key should not be present in the result"
        )
        os.unlink(self.tmp_model.name)

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
