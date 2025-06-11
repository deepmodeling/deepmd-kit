# SPDX-License-Identifier: LGPL-3.0-or-later
import io
import json
import os
import shutil
import unittest
from contextlib import (
    redirect_stderr,
)
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)

from .common import (
    run_dp,
)
from .model.test_permutation import (
    model_se_e2_a,
)


class TestSingleTaskModel(unittest.TestCase):
    def setUp(self) -> None:
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
        run_dp("dp --pt freeze")

    def test_checkpoint(self) -> None:
        INPUT = "model.pt"
        ATTRIBUTES = "type-map descriptor fitting-net size"
        with redirect_stderr(io.StringIO()) as f:
            run_dp(f"dp --pt show {INPUT} {ATTRIBUTES}")
        results = f.getvalue().split("\n")[:-1]
        assert "This is a singletask model" in results[-8]
        assert "The type_map is ['O', 'H', 'Au']" in results[-7]
        assert (
            "{'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut': 4.0"
        ) in results[-6]
        assert (
            "The fitting_net parameter is {'neuron': [24, 24, 24], 'resnet_dt': True, 'seed': 1}"
            in results[-5]
        )
        assert "Parameter counts:" in results[-4]
        assert "Parameters in descriptor: 19,350" in results[-3]
        assert "Parameters in fitting-net: 119,091" in results[-2]
        assert "Parameters in total: 138,441" in results[-1]

    def test_frozen_model(self) -> None:
        INPUT = "frozen_model.pth"
        ATTRIBUTES = "type-map descriptor fitting-net size"
        with redirect_stderr(io.StringIO()) as f:
            run_dp(f"dp --pt show {INPUT} {ATTRIBUTES}")
        results = f.getvalue().split("\n")[:-1]
        assert "This is a singletask model" in results[-8]
        assert "The type_map is ['O', 'H', 'Au']" in results[-7]
        assert (
            "{'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut': 4.0"
        ) in results[-6]
        assert (
            "The fitting_net parameter is {'neuron': [24, 24, 24], 'resnet_dt': True, 'seed': 1}"
            in results[-5]
        )
        assert "Parameter counts:" in results[-4]
        assert "Parameters in descriptor: 19,350" in results[-3]
        assert "Parameters in fitting-net: 119,091" in results[-2]
        assert "Parameters in total: 138,441" in results[-1]

    def test_checkpoint_error(self) -> None:
        INPUT = "model.pt"
        ATTRIBUTES = "model-branch type-map descriptor fitting-net"
        with self.assertRaisesRegex(
            RuntimeError, "The 'model-branch' option requires a multitask model"
        ):
            run_dp(f"dp --pt show {INPUT} {ATTRIBUTES}")

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth", "output.txt", "checkpoint"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestMultiTaskModel(unittest.TestCase):
    def setUp(self) -> None:
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
        run_dp("dp --pt freeze --model-branch model_1")

    def test_checkpoint(self) -> None:
        INPUT = "model.ckpt.pt"
        ATTRIBUTES = "model-branch type-map descriptor fitting-net size"
        with redirect_stderr(io.StringIO()) as f:
            run_dp(f"dp --pt show {INPUT} {ATTRIBUTES}")
        results = f.getvalue().split("\n")[:-1]
        assert "This is a multitask model" in results[-12]
        assert (
            "Available model branches are ['model_1', 'model_2', 'RANDOM'], "
            "where 'RANDOM' means using a randomly initialized fitting net."
            in results[-11]
        )
        assert "The type_map of branch model_1 is ['O', 'H', 'B']" in results[-10]
        assert "The type_map of branch model_2 is ['O', 'H', 'B']" in results[-9]
        assert (
            "model_1"
            and "'type': 'se_e2_a'"
            and "'sel': [46, 92, 4]"
            and "'rcut_smth': 0.5"
        ) in results[-8]
        assert (
            "model_2"
            and "'type': 'se_e2_a'"
            and "'sel': [46, 92, 4]"
            and "'rcut_smth': 0.5"
        ) in results[-7]
        assert (
            "The fitting_net parameter of branch model_1 is {'neuron': [1, 2, 3], 'seed': 678}"
            in results[-6]
        )
        assert (
            "The fitting_net parameter of branch model_2 is {'neuron': [9, 8, 7], 'seed': 1111}"
            in results[-5]
        )
        assert "Parameter counts for a single branch model:" in results[-4]
        assert "Parameters in descriptor: 19,350" in results[-3]
        assert "Parameters in fitting-net: 4,860" in results[-2]
        assert "Parameters in total: 24,210" in results[-1]

    def test_frozen_model(self) -> None:
        INPUT = "frozen_model.pth"
        ATTRIBUTES = "type-map descriptor fitting-net size"
        with redirect_stderr(io.StringIO()) as f:
            run_dp(f"dp --pt show {INPUT} {ATTRIBUTES}")
        results = f.getvalue().split("\n")[:-1]
        assert "This is a singletask model" in results[-8]
        assert "The type_map is ['O', 'H', 'B']" in results[-7]
        assert (
            "'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut_smth': 0.5"
        ) in results[-6]
        assert (
            "The fitting_net parameter is {'neuron': [1, 2, 3], 'seed': 678}"
            in results[-5]
        )
        assert "Parameter counts:" in results[-4]
        assert "Parameters in descriptor: 19,350" in results[-3]
        assert "Parameters in fitting-net: 4,860" in results[-2]
        assert "Parameters in total: 24,210" in results[-1]

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth", "checkpoint", "output.txt"]:
                os.remove(f)
            if f in ["stat_files", self.stat_files]:
                shutil.rmtree(f)
