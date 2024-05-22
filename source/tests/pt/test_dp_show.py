# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import unittest
from copy import (
    deepcopy,
)
from typing import ClassVar
from pathlib import (
    Path,
)

import torch

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)

from .model.test_permutation import (
    model_se_e2_a,
)


# mock FLAGS object
class FLAGS:
    INPUT: ClassVar[str] = ""
    ATTRIBUTES: ClassVar[list] = []


# mock log
class MockLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


# mock show function, the only difference from origin function is
# add `log = MockLogger()` and `return log.messages`
def mock_show(FLAGS):
    log = MockLogger()
    if FLAGS.INPUT.split(".")[-1] == "pt":
        state_dict = torch.load(FLAGS.INPUT, map_location=env.DEVICE)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_params = state_dict["_extra_state"]["model_params"]
    elif FLAGS.INPUT.split(".")[-1] == "pth":
        model_params_string = torch.jit.load(
            FLAGS.INPUT, map_location=env.DEVICE
        ).model_def_script
        model_params = json.loads(model_params_string)
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint file with a .pt extension "
            "or a frozen model with a .pth extension"
        )
    model_is_multi_task = "model_dict" in model_params
    log.info("This is a multitask model") if model_is_multi_task else log.info(
        "This is a singletask model"
    )

    if "model-branch" in FLAGS.ATTRIBUTES:
        #  The model must be multitask mode
        if not model_is_multi_task:
            raise RuntimeError(
                "The 'model-branch' option requires a multitask model."
                " The provided model does not meet this criterion."
            )
        model_branches = list(model_params["model_dict"].keys())
        log.info(f"Available model branches are {model_branches}")
    if "type-map" in FLAGS.ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                type_map = model_params["model_dict"][branch]["type_map"]
                log.info(f"The type_map of branch {branch} is {type_map}")
        else:
            type_map = model_params["type_map"]
            log.info(f"The type_map is {type_map}")
    if "descriptor" in FLAGS.ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                descriptor = model_params["model_dict"][branch]["descriptor"]
                log.info(f"The descriptor parameter of branch {branch} is {descriptor}")
        else:
            descriptor = model_params["descriptor"]
            log.info(f"The descriptor parameter is {descriptor}")
    if "fitting-net" in FLAGS.ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                fitting_net = model_params["model_dict"][branch]["fitting_net"]
                log.info(
                    f"The fitting_net parameter of branch {branch} is {fitting_net}"
                )
        else:
            fitting_net = model_params["fitting_net"]
            log.info(f"The fitting_net parameter is {fitting_net}")

    return log.messages


class TestSingleTaskModel(unittest.TestCase):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["model"]["type_map"] = ["O", "H", "Au"]
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        os.system("dp --pt freeze")

    def test_checkpoint(self):
        flags = FLAGS
        flags.INPUT = "model.pt"
        flags.ATTRIBUTES = ["type-map", "descriptor", "fitting-net"]
        log_messages = mock_show(flags)
        assert len(log_messages) == 4
        assert log_messages[0] == "This is a singletask model"
        assert log_messages[1] == "The type_map is ['O', 'H', 'Au']"
        assert (
            "{'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut': 4.0"
        ) in log_messages[2]
        assert (
            log_messages[3]
            == "The fitting_net parameter is {'neuron': [24, 24, 24], 'resnet_dt': True, 'seed': 1}"
        )

    def test_frozen_model(self):
        flags = FLAGS
        flags.INPUT = "frozen_model.pth"
        flags.ATTRIBUTES = ["type-map", "descriptor", "fitting-net"]
        log_messages = mock_show(flags)
        assert len(log_messages) == 4
        assert log_messages[0] == "This is a singletask model"
        assert log_messages[1] == "The type_map is ['O', 'H', 'Au']"
        assert (
            "{'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut': 4.0"
        ) in log_messages[2]
        assert (
            log_messages[3]
            == "The fitting_net parameter is {'neuron': [24, 24, 24], 'resnet_dt': True, 'seed': 1}"
        )

    def test_checkpoint_error(self):
        flags = FLAGS
        flags.INPUT = "model.pt"
        flags.ATTRIBUTES = ["model-branch", "type-map", "descriptor", "fitting-net"]
        with self.assertRaisesRegex(
            RuntimeError, "The 'model-branch' option requires a multitask model"
        ):
            log_messages = mock_show(flags)

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
            if f in ["checkpoint"]:
                os.remove(f)


class TestMultiTaskModel(unittest.TestCase):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/multitask.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["model"]["shared_dict"]["my_descriptor"] = model_se_e2_a[
            "descriptor"
        ]
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.stat_files = "se_e2_a"
        os.makedirs(self.stat_files, exist_ok=True)
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
        self.config["model"]["model_dict"]["model_1"]["fitting_net"] = {
            "neuron": [1, 2, 3],
            "seed": 678,
        }
        self.config["model"]["model_dict"]["model_2"]["fitting_net"] = {
            "neuron": [9, 8, 7],
            "seed": 1111,
        }
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.origin_config = deepcopy(self.config)
        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )
        trainer = get_trainer(deepcopy(self.config), shared_links=self.shared_links)
        trainer.run()
        os.system("dp --pt freeze --head model_1")

    def test_checkpoint(self):
        flags = FLAGS
        flags.INPUT = "model.ckpt.pt"
        flags.ATTRIBUTES = ["model-branch", "type-map", "descriptor", "fitting-net"]
        log_messages = mock_show(flags)
        assert len(log_messages) == 8
        assert log_messages[0] == "This is a multitask model"
        assert log_messages[1] == "Available model branches are ['model_1', 'model_2']"
        assert log_messages[2] == "The type_map of branch model_1 is ['O', 'H', 'B']"
        assert log_messages[3] == "The type_map of branch model_2 is ['O', 'H', 'B']"
        assert (
            "model_1"
            and "'type': 'se_e2_a'"
            and "'sel': [46, 92, 4]"
            and "'rcut_smth': 0.5"
        ) in log_messages[4]
        assert (
            "model_2"
            and "'type': 'se_e2_a'"
            and "'sel': [46, 92, 4]"
            and "'rcut_smth': 0.5"
        ) in log_messages[5]
        assert (
            log_messages[6]
            == "The fitting_net parameter of branch model_1 is {'neuron': [1, 2, 3], 'seed': 678}"
        )
        assert (
            log_messages[7]
            == "The fitting_net parameter of branch model_2 is {'neuron': [9, 8, 7], 'seed': 1111}"
        )

    def test_frozen_model(self):
        flags = FLAGS
        flags.INPUT = "frozen_model.pth"
        flags.ATTRIBUTES = ["type-map", "descriptor", "fitting-net"]
        log_messages = mock_show(flags)
        assert len(log_messages) == 4
        assert log_messages[0] == "This is a singletask model"
        assert log_messages[1] == "The type_map is ['O', 'H', 'B']"
        assert (
            "'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut_smth': 0.5"
        ) in log_messages[2]
        assert (
            log_messages[3]
            == "The fitting_net parameter is {'neuron': [1, 2, 3], 'seed': 678}"
        )

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
            if f in ["checkpoint"]:
                os.remove(f)
            if f in [self.stat_files]:
                shutil.rmtree(f)
