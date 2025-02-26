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
from deepmd.utils.data import (
    DeepmdData,
)

import torch

from deepmd.entrypoints.test import test_ener as dp_test_ener
from deepmd.pt.entrypoints.main import (
    get_trainer,
)

from deepmd.infer.deep_eval import (
    DeepEval,
)

class Test_testener(unittest.TestCase):
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
        model_se_e2_a= {
                        "type_map": ["O", "H", "B"],
                        "descriptor": {
                            "type": "se_e2_a",
                            "sel": [46, 92, 4],
                            "rcut_smth": 0.50,
                            "rcut": 4.00,
                            "neuron": [25, 50, 100],
                            "resnet_dt": False,
                            "axis_neuron": 16,
                            "seed": 1,
                        },
                        "fitting_net": {
                            "neuron": [24, 24, 24],
                            "resnet_dt": True,
                            "seed": 1,
                        },
                        "data_stat_nbatch": 20,
                    }
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, _, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        model = torch.jit.script(trainer.model)
        self.tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, self.tmp_model.name)

    def test_dp_test_1_frame(self) -> None:
        dp = DeepEval(self.tmp_model.name,head='PyTorch')
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
                system=self.config["training"]["validation_data"]["systems"],
                numb_test=1,
                detail_file=None,
                has_atom_ener=False,
            )
        print(err)
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

