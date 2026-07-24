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
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
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


class TestChangeBiasMultitask(unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/multitask.json")
        with open(input_json) as f:
            config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "change-bias-multitask-stat"
        os.makedirs(self.stat_files, exist_ok=True)
        config["model"]["shared_dict"]["my_descriptor"] = deepcopy(
            model_se_e2_a["descriptor"]
        )
        for model_key in config["training"]["data_dict"]:
            config["training"]["data_dict"][model_key]["training_data"]["systems"] = (
                data_file
            )
            config["training"]["data_dict"][model_key]["validation_data"]["systems"] = (
                data_file
            )
            config["training"]["data_dict"][model_key]["stat_file"] = (
                f"{self.stat_files}/{model_key}"
            )
        config["training"]["numb_steps"] = 0
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=True)
        config = normalize(config, multi_task=True)
        self.trainer = get_trainer(deepcopy(config), shared_links=shared_links)
        self.model_path = Path(current_path) / "change-bias-multitask-model.pt"
        self.model_path_user_bias = (
            Path(current_path) / "change-bias-multitask-model-user-bias.pt"
        )
        torch.save({"model": self.trainer.wrapper.state_dict()}, self.model_path)

    @staticmethod
    def _share_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()

    def _find_shared_descriptor_pair(
        self, state_dict: dict[str, torch.Tensor]
    ) -> tuple[str, str]:
        for key, value in state_dict.items():
            if not (
                key.startswith("model.model_1.")
                and "descriptor" in key
                and torch.is_tensor(value)
            ):
                continue
            peer_key = key.replace("model.model_1.", "model.model_2.", 1)
            if (
                peer_key in state_dict
                and torch.is_tensor(state_dict[peer_key])
                and self._share_storage(value, state_dict[peer_key])
            ):
                return key, peer_key
        self.fail("No shared descriptor tensor pair found in multitask checkpoint.")

    def test_change_bias_preserves_shared_checkpoint_storage(self) -> None:
        state_dict = torch.load(
            str(self.model_path), map_location=DEVICE, weights_only=True
        )["model"]
        shared_key, peer_key = self._find_shared_descriptor_pair(state_dict)

        user_bias = [0.1, 0.2, 0.3]
        run_dp(
            f"dp --pt change-bias {self.model_path!s} --model-branch model_1 "
            f"-b {' '.join([str(_) for _ in user_bias])} "
            f"-o {self.model_path_user_bias!s}"
        )
        updated_state_dict = torch.load(
            str(self.model_path_user_bias), map_location=DEVICE, weights_only=True
        )["model"]

        self.assertTrue(
            self._share_storage(
                updated_state_dict[shared_key], updated_state_dict[peer_key]
            )
        )
        bias_keys = [
            key
            for key in updated_state_dict
            if key.startswith("model.model_1.") and key.endswith("out_bias")
        ]
        self.assertEqual(len(bias_keys), 1)
        updated_bias = updated_state_dict[bias_keys[0]]
        expected_bias = to_torch_tensor(np.array(user_bias)).view(updated_bias.shape)
        torch.testing.assert_close(updated_bias, expected_bias)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("change-bias-multitask-model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in [self.stat_files]:
                shutil.rmtree(f)
