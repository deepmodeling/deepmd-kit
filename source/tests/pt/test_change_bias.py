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

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.train.training import (
    get_model_for_wrapper,
    model_change_out_bias,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)

from .common import (
    run_dp,
)
from .model.test_permutation import (
    model_se_e2_a,
)
from .test_finetune import (
    energy_data_requirement,
)

current_path = os.getcwd()


class TestChangeBias(unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        model_name = "change-bias-model.ckpt"
        self.data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["save_ckpt"] = model_name
        self.trainer = get_trainer(deepcopy(self.config))
        self.trainer.run()
        self.state_dict_trained = self.trainer.wrapper.model.state_dict()
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(energy_data_requirement)
        self.sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )
        self.model_path = Path(current_path) / (model_name + ".pt")
        self.model_path_data_bias = Path(current_path) / (
            model_name + "data_bias" + ".pt"
        )
        self.model_path_data_file_bias = Path(current_path) / (
            model_name + "data_file_bias" + ".pt"
        )
        self.model_path_user_bias = Path(current_path) / (
            model_name + "user_bias" + ".pt"
        )
        self.loss_params = self.config["loss"]

    def test_change_bias_with_data(self) -> None:
        run_dp(
            f"dp --pt change-bias {self.model_path!s} -s {self.data_file[0]} -o {self.model_path_data_bias!s}"
        )
        state_dict = torch.load(
            str(self.model_path_data_bias), map_location=DEVICE, weights_only=True
        )
        model_params = state_dict["model"]["_extra_state"]["model_params"]
        model_for_wrapper = get_model_for_wrapper(
            model_params,
            _loss_params=self.loss_params,
        )
        wrapper = ModelWrapper(model_for_wrapper)
        wrapper.load_state_dict(state_dict["model"])
        updated_bias = wrapper.model["Default"].get_out_bias()
        expected_model = model_change_out_bias(
            self.trainer.wrapper.model["Default"],
            self.sampled,
            _bias_adjust_mode="change-by-statistic",
        )
        expected_bias = expected_model.get_out_bias()
        torch.testing.assert_close(updated_bias, expected_bias)

    def test_change_bias_with_data_sys_file(self) -> None:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        with open(tmp_file.name, "w") as f:
            f.writelines([sys + "\n" for sys in self.data_file])
        run_dp(
            f"dp --pt change-bias {self.model_path!s} -f {tmp_file.name} -o {self.model_path_data_file_bias!s}"
        )
        state_dict = torch.load(
            str(self.model_path_data_file_bias), map_location=DEVICE, weights_only=True
        )
        model_params = state_dict["model"]["_extra_state"]["model_params"]
        model_for_wrapper = get_model_for_wrapper(
            model_params,
            _loss_params=self.loss_params,
        )
        wrapper = ModelWrapper(model_for_wrapper)
        wrapper.load_state_dict(state_dict["model"])
        updated_bias = wrapper.model["Default"].get_out_bias()
        expected_model = model_change_out_bias(
            self.trainer.wrapper.model["Default"],
            self.sampled,
            _bias_adjust_mode="change-by-statistic",
        )
        expected_bias = expected_model.get_out_bias()
        torch.testing.assert_close(updated_bias, expected_bias)

    def test_change_bias_with_user_defined(self) -> None:
        user_bias = [0.1, 3.2, -0.5]
        run_dp(
            f"dp --pt change-bias {self.model_path!s} -b {' '.join([str(_) for _ in user_bias])} -o {self.model_path_user_bias!s}"
        )
        state_dict = torch.load(
            str(self.model_path_user_bias), map_location=DEVICE, weights_only=True
        )
        model_params = state_dict["model"]["_extra_state"]["model_params"]
        model_for_wrapper = get_model_for_wrapper(
            model_params,
            _loss_params=self.loss_params,
        )
        wrapper = ModelWrapper(model_for_wrapper)
        wrapper.load_state_dict(state_dict["model"])
        updated_bias = wrapper.model["Default"].get_out_bias()
        expected_bias = to_torch_tensor(np.array(user_bias)).view(updated_bias.shape)
        torch.testing.assert_close(updated_bias, expected_bias)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("change-bias-model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
