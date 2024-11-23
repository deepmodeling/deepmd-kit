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

import torch

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.infer import (
    inference,
)

from .test_permutation import (
    model_dos,
    model_dpa1,
    model_dpa2,
    model_hybrid,
    model_se_e2_a,
)


class JITTest:
    def test_jit(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        model = torch.jit.script(inference.Tester("./model.pt").model)
        torch.jit.save(model, "./frozen_model.pth", {})

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
            if f in ["checkpoint"]:
                os.remove(f)


class TestEnergyModelSeA(unittest.TestCase, JITTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self) -> None:
        JITTest.tearDown(self)


class TestDOSModelSeA(unittest.TestCase, JITTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent.parent / "dos/input.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent.parent / "dos/data/global_system")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dos)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self) -> None:
        JITTest.tearDown(self)


class TestEnergyModelDPA1(unittest.TestCase, JITTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dpa1)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self) -> None:
        JITTest.tearDown(self)


class TestEnergyModelDPA2(unittest.TestCase, JITTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dpa2)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self) -> None:
        JITTest.tearDown(self)


class TestEnergyModelHybrid(unittest.TestCase, JITTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_hybrid)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self) -> None:
        JITTest.tearDown(self)


class TestEnergyModelHybrid2(unittest.TestCase, JITTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_hybrid)
        # self.config["model"]["descriptor"]["hybrid_mode"] = "sequential"
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self) -> None:
        JITTest.tearDown(self)


class TestEnergyModelDPA2IntRcut(unittest.TestCase, JITTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dpa2)
        self.config["model"]["descriptor"]["repinit"]["rcut"] = int(
            self.config["model"]["descriptor"]["repinit"]["rcut"]
        )
        self.config["model"]["descriptor"]["repinit"]["rcut_smth"] = int(
            self.config["model"]["descriptor"]["repinit"]["rcut_smth"]
        )
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self) -> None:
        JITTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
