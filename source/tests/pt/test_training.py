# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

from deepmd.pt.entrypoints.main import (
    get_trainer,
)

from .test_permutation import (
    model_dpa1,
    model_dpa2,
    model_hybrid,
    model_se_e2_a,
)


class DPTrainTest:
    def test_dp_train(self):
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        self.tearDown()

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestEnergyModelSeA(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestEnergyModelDPA1(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dpa1)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestEnergyModelDPA2(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dpa2)
        self.config["model"]["descriptor"]["rcut"] = self.config["model"]["descriptor"][
            "repinit_rcut"
        ]
        self.config["model"]["descriptor"]["rcut_smth"] = self.config["model"][
            "descriptor"
        ]["repinit_rcut_smth"]
        self.config["model"]["descriptor"]["sel"] = self.config["model"]["descriptor"][
            "repinit_nsel"
        ]
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


@unittest.skip("hybrid not supported at the moment")
class TestEnergyModelHybrid(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_hybrid)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
