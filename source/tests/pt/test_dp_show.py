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

from .model.test_permutation import (
    model_se_e2_a,
)


def read_output_file(file_path):
    with open(file_path) as f:
        return f.readlines()


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
        INPUT = "model.pt"
        ATTRIBUTES = "type-map descriptor fitting-net"
        os.system(f"dp --pt show {INPUT} {ATTRIBUTES} 2> output.txt")
        results = read_output_file("output.txt")
        assert "This is a singletask model" in results[-4]
        assert "The type_map is ['O', 'H', 'Au']" in results[-3]
        assert (
            "{'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut': 4.0"
        ) in results[-2]
        assert (
            "The fitting_net parameter is {'neuron': [24, 24, 24], 'resnet_dt': True, 'seed': 1}"
            in results[-1]
        )

    def test_frozen_model(self):
        INPUT = "frozen_model.pth"
        ATTRIBUTES = "type-map descriptor fitting-net"
        os.system(f"dp --pt show {INPUT} {ATTRIBUTES} 2> output.txt")
        results = read_output_file("output.txt")
        assert "This is a singletask model" in results[-4]
        assert "The type_map is ['O', 'H', 'Au']" in results[-3]
        assert (
            "{'type': 'se_e2_a'" and "'sel': [46, 92, 4]" and "'rcut': 4.0"
        ) in results[-2]
        assert (
            "The fitting_net parameter is {'neuron': [24, 24, 24], 'resnet_dt': True, 'seed': 1}"
            in results[-1]
        )

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth", "output.txt", "checkpoint"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
