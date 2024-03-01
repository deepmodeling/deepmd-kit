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
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)

from .model.test_permutation import (
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
)

multitask_template_json = str(Path(__file__).parent / "water/multitask.json")
with open(multitask_template_json) as f:
    multitask_template = json.load(f)


class MultiTaskTrainTest:
    def test_multitask_train(self):
        trainer = get_trainer(deepcopy(self.config), shared_links=self.shared_links)
        trainer.run()
        # check model keys
        self.assertEqual(len(trainer.wrapper.model), 2)
        self.assertIn("model_1", trainer.wrapper.model)
        self.assertIn("model_2", trainer.wrapper.model)

        # check shared parameters
        multi_state_dict = trainer.wrapper.model.state_dict()
        for state_key in multi_state_dict:
            if "model_1" in state_key:
                self.assertIn(state_key.replace("model_1", "model_2"), multi_state_dict)
            if "model_2" in state_key:
                self.assertIn(state_key.replace("model_2", "model_1"), multi_state_dict)
            if "model_1.descriptor" in state_key:
                torch.testing.assert_allclose(
                    multi_state_dict[state_key],
                    multi_state_dict[state_key.replace("model_1", "model_2")],
                )
        self.tearDown()

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in [self.stat_files]:
                shutil.rmtree(f)


class TestMultiTaskSeA(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self):
        multitask_se_e2_a = deepcopy(multitask_template)
        multitask_se_e2_a["model"]["shared_dict"]["my_descriptor"] = model_se_e2_a[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "se_e2_a"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_se_e2_a
        self.config["training"]["data_dict"]["model_1"]["training_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"][
            "stat_file"
        ] = f"{self.stat_files}/model_1"
        self.config["training"]["data_dict"]["model_2"]["training_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"][
            "stat_file"
        ] = f"{self.stat_files}/model_2"
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


class TestMultiTaskDPA1(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self):
        multitask_DPA1 = deepcopy(multitask_template)
        multitask_DPA1["model"]["shared_dict"]["my_descriptor"] = model_dpa1[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "DPA1"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_DPA1
        self.config["training"]["data_dict"]["model_1"]["training_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"][
            "stat_file"
        ] = f"{self.stat_files}/model_1"
        self.config["training"]["data_dict"]["model_2"]["training_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"][
            "stat_file"
        ] = f"{self.stat_files}/model_2"
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


class TestMultiTaskDPA2(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self):
        multitask_DPA2 = deepcopy(multitask_template)
        multitask_DPA2["model"]["shared_dict"]["my_descriptor"] = model_dpa2[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "DPA2"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_DPA2
        self.config["training"]["data_dict"]["model_1"]["training_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"][
            "stat_file"
        ] = f"{self.stat_files}/model_1"
        self.config["training"]["data_dict"]["model_2"]["training_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"][
            "stat_file"
        ] = f"{self.stat_files}/model_2"
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
