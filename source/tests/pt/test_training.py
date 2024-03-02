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

from .model.test_permutation import (
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

    def test_trainable(self):
        fix_params = deepcopy(self.config)
        fix_params["model"]["descriptor"]["trainable"] = False
        fix_params["model"]["fitting_net"]["trainable"] = False
        trainer_fix = get_trainer(fix_params)
        model_dict_before_training = deepcopy(trainer_fix.model.state_dict())
        trainer_fix.run()
        model_dict_after_training = deepcopy(trainer_fix.model.state_dict())
        for key in model_dict_before_training:
            torch.testing.assert_allclose(
                model_dict_before_training[key], model_dict_after_training[key]
            )
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


class TestDipoleModelSeA(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water_tensor/se_e2_a.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file_atomic = str(
            Path(__file__).parent / "water_tensor/dipole/atomic_system"
        )
        data_file_global = str(
            Path(__file__).parent / "water_tensor/dipole/global_system"
        )
        self.config["training"]["training_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["training"]["validation_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["model"]["atom_exclude_types"] = [1]
        self.config["model"]["fitting_net"]["type"] = "dipole"
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestDipoleModelDPA1(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water_tensor/se_e2_a.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file_atomic = str(
            Path(__file__).parent / "water_tensor/dipole/atomic_system"
        )
        data_file_global = str(
            Path(__file__).parent / "water_tensor/dipole/global_system"
        )
        self.config["training"]["training_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["training"]["validation_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["model"] = deepcopy(model_dpa1)
        self.config["model"]["atom_exclude_types"] = [1]
        self.config["model"]["fitting_net"]["type"] = "dipole"
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestDipoleModelDPA2(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water_tensor/se_e2_a.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file_atomic = str(
            Path(__file__).parent / "water_tensor/dipole/atomic_system"
        )
        data_file_global = str(
            Path(__file__).parent / "water_tensor/dipole/global_system"
        )
        self.config["training"]["training_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["training"]["validation_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["model"] = deepcopy(model_dpa2)
        self.config["model"]["atom_exclude_types"] = [1]
        self.config["model"]["fitting_net"]["type"] = "dipole"
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestPolarModelSeA(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water_tensor/se_e2_a.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file_atomic = str(
            Path(__file__).parent / "water_tensor/polar/atomic_system"
        )
        data_file_global = str(
            Path(__file__).parent / "water_tensor/polar/global_system"
        )
        self.config["training"]["training_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["training"]["validation_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["model"]["atom_exclude_types"] = [1]
        self.config["model"]["fitting_net"]["type"] = "polar"
        self.config["model"]["fitting_net"]["fit_diag"] = False
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestPolarModelDPA1(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water_tensor/se_e2_a.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file_atomic = str(
            Path(__file__).parent / "water_tensor/polar/atomic_system"
        )
        data_file_global = str(
            Path(__file__).parent / "water_tensor/polar/global_system"
        )
        self.config["training"]["training_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["training"]["validation_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["model"] = deepcopy(model_dpa1)
        self.config["model"]["atom_exclude_types"] = [1]
        self.config["model"]["fitting_net"]["type"] = "polar"
        self.config["model"]["fitting_net"]["fit_diag"] = False
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestPolarModelDPA2(unittest.TestCase, DPTrainTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water_tensor/se_e2_a.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file_atomic = str(
            Path(__file__).parent / "water_tensor/polar/atomic_system"
        )
        data_file_global = str(
            Path(__file__).parent / "water_tensor/polar/global_system"
        )
        self.config["training"]["training_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["training"]["validation_data"]["systems"] = [
            data_file_atomic,
            data_file_global,
        ]
        self.config["model"] = deepcopy(model_dpa2)
        self.config["model"]["atom_exclude_types"] = [1]
        self.config["model"]["fitting_net"]["type"] = "polar"
        self.config["model"]["fitting_net"]["fit_diag"] = False
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
