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

import torch

from deepmd.entrypoints.test import test as dp_test
from deepmd.pt_expt.entrypoints.main import (
    freeze,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)

model_se_e2_a = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
}


class TestDPTestPtExpt(unittest.TestCase):
    """Test dp test for the pt_expt backend (.pte models)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_file = str(
            Path(__file__).parents[1] / "pt" / "water" / "data" / "single"
        )
        cls.detail_file = os.path.join(
            tempfile.mkdtemp(), "test_dp_test_pt_expt_detail"
        )
        cls.tmpdir = tempfile.mkdtemp()

        # Build a model, save a checkpoint, and freeze to .pte
        model_params = deepcopy(model_se_e2_a)
        model = get_model(model_params)
        wrapper = ModelWrapper(model, model_params=model_params)
        state_dict = wrapper.state_dict()
        ckpt_file = os.path.join(cls.tmpdir, "model.pt")
        torch.save({"model": state_dict}, ckpt_file)

        cls.pte_file = os.path.join(cls.tmpdir, "frozen_model.pte")
        freeze(model=ckpt_file, output=cls.pte_file)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)
        detail_dir = os.path.dirname(cls.detail_file)
        if os.path.exists(detail_dir):
            shutil.rmtree(detail_dir)

    def test_dp_test_system(self) -> None:
        """Test dp test with -s system path."""
        detail = self.detail_file + "_sys"
        dp_test(
            model=self.pte_file,
            system=self.data_file,
            datafile=None,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail,
            atomic=False,
        )
        self.assertTrue(os.path.exists(detail + ".e.out"))
        self.assertTrue(os.path.exists(detail + ".f.out"))
        self.assertTrue(os.path.exists(detail + ".v.out"))

    def test_dp_test_input_json(self) -> None:
        """Test dp test with --valid-data JSON input."""
        config = {
            "model": deepcopy(model_se_e2_a),
            "training": {
                "training_data": {"systems": [self.data_file]},
                "validation_data": {"systems": [self.data_file]},
            },
        }
        input_json = os.path.join(self.tmpdir, "test_input.json")
        with open(input_json, "w") as fp:
            json.dump(config, fp, indent=4)

        detail = self.detail_file + "_json"
        dp_test(
            model=self.pte_file,
            system=None,
            datafile=None,
            valid_json=input_json,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail,
            atomic=False,
        )
        self.assertTrue(os.path.exists(detail + ".e.out"))


if __name__ == "__main__":
    unittest.main()
