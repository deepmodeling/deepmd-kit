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

import numpy as np

from deepmd.pd.entrypoints.main import (
    get_trainer,
)
from deepmd.pd.utils.env import (
    enable_prim,
)
from deepmd.pd.utils.finetune import (
    get_finetune_rules,
)

from .model.test_permutation import (
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
)


class DPTrainTest:
    def test_dp_train(self) -> None:
        # test training from scratch
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        state_dict_trained = trainer.wrapper.model.state_dict()

        # test fine-tuning using same input
        finetune_model = self.config["training"].get("save_ckpt", "model.ckpt") + ".pd"
        self.config["model"], finetune_links = get_finetune_rules(
            finetune_model,
            self.config["model"],
        )
        trainer_finetune = get_trainer(
            deepcopy(self.config),
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # test fine-tuning using empty input
        self.config_empty = deepcopy(self.config)
        if "descriptor" in self.config_empty["model"]:
            self.config_empty["model"]["descriptor"] = {}
        if "fitting_net" in self.config_empty["model"]:
            self.config_empty["model"]["fitting_net"] = {}
        self.config_empty["model"], finetune_links = get_finetune_rules(
            finetune_model,
            self.config_empty["model"],
            change_model_params=True,
        )
        trainer_finetune_empty = get_trainer(
            deepcopy(self.config_empty),
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # test fine-tuning using random fitting
        self.config["model"], finetune_links = get_finetune_rules(
            finetune_model, self.config["model"], model_branch="RANDOM"
        )
        trainer_finetune_random = get_trainer(
            deepcopy(self.config_empty),
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # check parameters
        state_dict_finetuned = trainer_finetune.wrapper.model.state_dict()
        state_dict_finetuned_empty = trainer_finetune_empty.wrapper.model.state_dict()
        state_dict_finetuned_random = trainer_finetune_random.wrapper.model.state_dict()
        for state_key in state_dict_finetuned:
            if "out_bias" not in state_key and "out_std" not in state_key:
                np.testing.assert_allclose(
                    state_dict_trained[state_key].numpy(),
                    state_dict_finetuned[state_key].numpy(),
                )
                np.testing.assert_allclose(
                    state_dict_trained[state_key].numpy(),
                    state_dict_finetuned_empty[state_key].numpy(),
                )
                if "fitting_net" not in state_key:
                    np.testing.assert_allclose(
                        state_dict_trained[state_key].numpy(),
                        state_dict_finetuned_random[state_key].numpy(),
                    )

        # check running
        trainer_finetune.run()
        trainer_finetune_empty.run()
        trainer_finetune_random.run()

    def test_trainable(self) -> None:
        fix_params = deepcopy(self.config)
        fix_params["model"]["descriptor"]["trainable"] = False
        fix_params["model"]["fitting_net"]["trainable"] = False
        free_descriptor = hasattr(self, "not_all_grad") and self.not_all_grad
        if free_descriptor:
            # can not set requires_grad false for all parameters,
            # because the input coord has no grad, thus the loss if all set to false
            # we only check trainable for fitting net
            fix_params["model"]["descriptor"]["trainable"] = True
            trainer_fix = get_trainer(fix_params)
            model_dict_before_training = deepcopy(
                trainer_fix.model.get_fitting_net().state_dict()
            )
            trainer_fix.run()
            model_dict_after_training = deepcopy(
                trainer_fix.model.get_fitting_net().state_dict()
            )
        else:
            trainer_fix = get_trainer(fix_params)
            model_dict_before_training = deepcopy(trainer_fix.model.state_dict())
            trainer_fix.run()
            model_dict_after_training = deepcopy(trainer_fix.model.state_dict())
        for key in model_dict_before_training:
            np.testing.assert_allclose(
                model_dict_before_training[key].numpy(),
                model_dict_after_training[key].numpy(),
            )

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pd"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestEnergyModelSeA(unittest.TestCase, DPTrainTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        # import paddle
        enable_prim(True)
        # assert paddle.framework.core._is_eager_prim_enabled()

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestFparam(unittest.TestCase, DPTrainTest):
    """Test if `fparam` can be loaded correctly."""

    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["model"]["fitting_net"]["numb_fparam"] = 1
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.set_path = Path(__file__).parent / "water/data/data_0" / "set.000"
        shutil.copyfile(self.set_path / "energy.npy", self.set_path / "fparam.npy")

    def tearDown(self) -> None:
        (self.set_path / "fparam.npy").unlink(missing_ok=True)
        DPTrainTest.tearDown(self)


class TestEnergyModelDPA1(unittest.TestCase, DPTrainTest):
    def setUp(self) -> None:
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
    def setUp(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
