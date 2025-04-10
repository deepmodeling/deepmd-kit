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
from deepmd.pt.utils.finetune import (
    get_finetune_rules,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

from .model.test_permutation import (
    model_dpa1,
    model_dpa2,
    model_dpa2tebd,
    model_dpa3,
    model_se_e2_a,
)


def setUpModule() -> None:
    global multitask_template
    multitask_template_json = str(Path(__file__).parent / "water/multitask.json")
    with open(multitask_template_json) as f:
        multitask_template = json.load(f)

    global multitask_sharefit_template
    multitask_sharefit_template_json = str(
        Path(__file__).parent / "water/multitask_sharefit.json"
    )
    with open(multitask_sharefit_template_json) as f:
        multitask_sharefit_template = json.load(f)


class MultiTaskTrainTest:
    def test_multitask_train(self) -> None:
        # test multitask training
        self.config = update_deepmd_input(self.config, warning=True)
        self.config = normalize(self.config, multi_task=True)
        self.share_fitting = getattr(self, "share_fitting", False)
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
            if ("model_1.atomic_model.descriptor" in state_key) or (
                self.share_fitting
                and "model_1.atomic_model.fitting_net" in state_key
                and "fitting_net.bias_atom_e" not in state_key
                and "fitting_net.case_embd" not in state_key
            ):
                torch.testing.assert_close(
                    multi_state_dict[state_key],
                    multi_state_dict[state_key.replace("model_1", "model_2")],
                )

        # test multitask fine-tuning
        # add model_3
        self.origin_config["model"]["model_dict"]["model_3"] = deepcopy(
            self.origin_config["model"]["model_dict"]["model_2"]
        )
        self.origin_config["loss_dict"]["model_3"] = deepcopy(
            self.origin_config["loss_dict"]["model_2"]
        )
        self.origin_config["training"]["model_prob"]["model_3"] = deepcopy(
            self.origin_config["training"]["model_prob"]["model_2"]
        )
        self.origin_config["training"]["data_dict"]["model_3"] = deepcopy(
            self.origin_config["training"]["data_dict"]["model_2"]
        )
        self.origin_config["training"]["data_dict"]["model_3"]["stat_file"] = (
            self.origin_config["training"]["data_dict"]["model_3"]["stat_file"].replace(
                "model_2", "model_3"
            )
        )

        # add model_4
        self.origin_config["model"]["model_dict"]["model_4"] = deepcopy(
            self.origin_config["model"]["model_dict"]["model_2"]
        )
        self.origin_config["loss_dict"]["model_4"] = deepcopy(
            self.origin_config["loss_dict"]["model_2"]
        )
        self.origin_config["training"]["model_prob"]["model_4"] = deepcopy(
            self.origin_config["training"]["model_prob"]["model_2"]
        )
        self.origin_config["training"]["data_dict"]["model_4"] = deepcopy(
            self.origin_config["training"]["data_dict"]["model_2"]
        )
        self.origin_config["training"]["data_dict"]["model_4"]["stat_file"] = (
            self.origin_config["training"]["data_dict"]["model_4"]["stat_file"].replace(
                "model_2", "model_4"
            )
        )

        # set finetune rules
        # model_1 resuming from model_1
        # pass

        # model_2 fine-tuning from model_2
        self.origin_config["model"]["model_dict"]["model_2"]["finetune_head"] = (
            "model_2"
        )

        # new model_3 fine-tuning from model_2
        self.origin_config["model"]["model_dict"]["model_3"]["finetune_head"] = (
            "model_2"
        )

        # new model_4 fine-tuning with randomly initialized fitting net
        # pass

        self.origin_config["model"], shared_links_finetune = preprocess_shared_params(
            self.origin_config["model"]
        )

        finetune_model = self.config["training"].get("save_ckpt", "model.ckpt") + ".pt"
        self.origin_config["model"], finetune_links = get_finetune_rules(
            finetune_model,
            self.origin_config["model"],
        )
        self.origin_config = update_deepmd_input(self.origin_config, warning=True)
        self.origin_config = normalize(self.origin_config, multi_task=True)
        trainer_finetune = get_trainer(
            deepcopy(self.origin_config),
            finetune_model=finetune_model,
            shared_links=shared_links_finetune,
            finetune_links=finetune_links,
        )

        # check parameters
        multi_state_dict_finetuned = trainer_finetune.wrapper.model.state_dict()
        for state_key in multi_state_dict_finetuned:
            if "model_1" in state_key:
                torch.testing.assert_close(
                    multi_state_dict[state_key],
                    multi_state_dict_finetuned[state_key],
                )
            elif "model_2" in state_key and "out_bias" not in state_key:
                torch.testing.assert_close(
                    multi_state_dict[state_key],
                    multi_state_dict_finetuned[state_key],
                )
            elif "model_3" in state_key and "out_bias" not in state_key:
                torch.testing.assert_close(
                    multi_state_dict[state_key.replace("model_3", "model_2")],
                    multi_state_dict_finetuned[state_key],
                )
            elif (
                "model_4" in state_key
                and "fitting_net" not in state_key
                and "out_bias" not in state_key
            ):
                torch.testing.assert_close(
                    multi_state_dict[state_key.replace("model_4", "model_2")],
                    multi_state_dict_finetuned[state_key],
                )

        # check running
        trainer_finetune.run()
        self.tearDown()

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in [self.stat_files]:
                shutil.rmtree(f)


class TestMultiTaskSeA(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self) -> None:
        multitask_se_e2_a = deepcopy(multitask_template)
        multitask_se_e2_a["model"]["shared_dict"]["my_descriptor"] = model_se_e2_a[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "se_e2_a"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_se_e2_a
        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["stat_file"] = (
            f"{self.stat_files}/model_1"
        )
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["stat_file"] = (
            f"{self.stat_files}/model_2"
        )
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.origin_config = deepcopy(self.config)
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


class TestMultiTaskSeASharefit(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self) -> None:
        multitask_se_e2_a = deepcopy(multitask_sharefit_template)
        multitask_se_e2_a["model"]["shared_dict"]["my_descriptor"] = model_se_e2_a[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "se_e2_a_share_fit"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_se_e2_a
        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["stat_file"] = (
            f"{self.stat_files}/model_1"
        )
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["stat_file"] = (
            f"{self.stat_files}/model_2"
        )
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.origin_config = deepcopy(self.config)
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )
        self.share_fitting = True

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


class TestMultiTaskDPA1(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self) -> None:
        multitask_DPA1 = deepcopy(multitask_template)
        multitask_DPA1["model"]["shared_dict"]["my_descriptor"] = model_dpa1[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "DPA1"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_DPA1
        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["stat_file"] = (
            f"{self.stat_files}/model_1"
        )
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["stat_file"] = (
            f"{self.stat_files}/model_2"
        )
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.origin_config = deepcopy(self.config)
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


class TestMultiTaskDPA2(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self) -> None:
        multitask_DPA2 = deepcopy(multitask_template)
        multitask_DPA2["model"]["shared_dict"]["my_descriptor"] = model_dpa2[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "DPA2"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_DPA2
        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["stat_file"] = (
            f"{self.stat_files}/model_1"
        )
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["stat_file"] = (
            f"{self.stat_files}/model_2"
        )
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.origin_config = deepcopy(self.config)
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


class TestMultiTaskDPA2Tebd(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self) -> None:
        multitask_DPA2 = deepcopy(multitask_template)
        multitask_DPA2["model"]["shared_dict"]["my_descriptor"] = model_dpa2tebd[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "DPA2Tebd"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_DPA2
        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["stat_file"] = (
            f"{self.stat_files}/model_1"
        )
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["stat_file"] = (
            f"{self.stat_files}/model_2"
        )
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.origin_config = deepcopy(self.config)
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


class TestMultiTaskDPA3(unittest.TestCase, MultiTaskTrainTest):
    def setUp(self) -> None:
        multitask_DPA3 = deepcopy(multitask_template)
        multitask_DPA3["model"]["shared_dict"]["my_descriptor"] = model_dpa3[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "DPA3"
        os.makedirs(self.stat_files, exist_ok=True)
        self.config = multitask_DPA3
        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_1"]["stat_file"] = (
            f"{self.stat_files}/model_1"
        )
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            data_file
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = data_file
        self.config["training"]["data_dict"]["model_2"]["stat_file"] = (
            f"{self.stat_files}/model_2"
        )
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.origin_config = deepcopy(self.config)
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )

    def tearDown(self) -> None:
        MultiTaskTrainTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
