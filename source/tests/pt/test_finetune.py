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

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.finetune import (
    get_finetune_rules,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .model.test_permutation import (
    model_dpa1,
)

energy_data_requirement = [
    DataRequirementItem(
        "energy",
        ndof=1,
        atomic=False,
        must=False,
        high_prec=True,
    ),
    DataRequirementItem(
        "force",
        ndof=3,
        atomic=True,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "virial",
        ndof=9,
        atomic=False,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "atom_ener",
        ndof=1,
        atomic=True,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "atom_pref",
        ndof=1,
        atomic=True,
        must=False,
        high_prec=False,
        repeat=3,
    ),
]


class FinetuneTest:
    def test_finetune_change_out_bias(self):
        # get data
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(energy_data_requirement)
        sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )
        # get model
        model = get_model(self.config["model"]).to(env.DEVICE)
        atomic_model = model.atomic_model
        atomic_model["out_bias"] = torch.rand_like(atomic_model["out_bias"])
        energy_bias_before = to_numpy_array(atomic_model["out_bias"])[0].ravel()

        # prepare original model for test
        dp = torch.jit.script(model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(dp, tmp_model.name)
        dp = DeepEval(tmp_model.name)
        origin_type_map = ["O", "H"]
        full_type_map = ["O", "H", "B"]

        # change energy bias
        model.atomic_model.change_out_bias(
            sampled,
            bias_adjust_mode="change-by-statistic",
        )
        energy_bias_after = to_numpy_array(atomic_model["out_bias"])[0].ravel()

        # get ground-truth energy bias change
        sorter = np.argsort(full_type_map)
        idx_type_map = sorter[
            np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
        ]
        ntest = 1
        atom_nums = np.tile(
            np.bincount(to_numpy_array(sampled[0]["atype"][0]))[idx_type_map],
            (ntest, 1),
        )
        energy = dp.eval(
            to_numpy_array(sampled[0]["coord"][:ntest]),
            to_numpy_array(sampled[0]["box"][:ntest]),
            to_numpy_array(sampled[0]["atype"][0]),
        )[0]
        energy_diff = to_numpy_array(sampled[0]["energy"][:ntest]) - energy
        finetune_shift = (
            energy_bias_after[idx_type_map] - energy_bias_before[idx_type_map]
        )
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        # check values
        np.testing.assert_almost_equal(finetune_shift, ground_truth_shift, decimal=10)

        self.tearDown()

    def test_finetune_slim_type(self):
        if not self.mixed_types:
            # skip when not mixed_types
            return
        # get data
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(energy_data_requirement)
        sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )
        data_type_map = self.config["model"]["type_map"]
        large_type_map = ["H", "X1", "X2", "O", "B"]
        large_type_map_index = np.array(
            [large_type_map.index(i) for i in data_type_map], dtype=np.int32
        )
        slimed_type_map = ["O", "H"]

        # get pretrained model with large type map
        config_large_type_map = deepcopy(self.config)
        config_large_type_map["model"]["type_map"] = large_type_map
        trainer = get_trainer(config_large_type_map)
        trainer.run()
        finetune_model = (
            config_large_type_map["training"].get("save_ckpt", "model.ckpt") + ".pt"
        )

        # finetune load the same type_map
        config_large_type_map_finetune = deepcopy(self.config)
        config_large_type_map_finetune["model"]["type_map"] = large_type_map
        config_large_type_map_finetune["model"], finetune_links = get_finetune_rules(
            finetune_model,
            config_large_type_map_finetune["model"],
        )
        trainer_finetune_large = get_trainer(
            config_large_type_map_finetune,
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # finetune load the slim type_map
        config_slimed_type_map_finetune = deepcopy(self.config)
        config_slimed_type_map_finetune["model"]["type_map"] = slimed_type_map
        config_slimed_type_map_finetune["model"], finetune_links = get_finetune_rules(
            finetune_model,
            config_slimed_type_map_finetune["model"],
        )
        trainer_finetune_slimed = get_trainer(
            config_slimed_type_map_finetune,
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # test consistency
        ntest = 1
        prec = 1e-10
        model_large_result = trainer_finetune_large.model(
            sampled[0]["coord"][:ntest],
            to_torch_tensor(large_type_map_index)[sampled[0]["atype"][:ntest]],
            box=sampled[0]["box"][:ntest],
        )
        model_slimed_result = trainer_finetune_slimed.model(
            sampled[0]["coord"][:ntest],
            sampled[0]["atype"][:ntest],
            box=sampled[0]["box"][:ntest],
        )
        test_keys = ["energy", "force", "virial"]
        for key in test_keys:
            torch.testing.assert_close(
                model_large_result[key],
                model_slimed_result[key],
                rtol=prec,
                atol=prec,
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


# class TestEnergyModelSeA(FinetuneTest, unittest.TestCase):
#     def setUp(self):
#         input_json = str(Path(__file__).parent / "water/se_atten.json")
#         with open(input_json) as f:
#             self.config = json.load(f)
#         self.data_file = [str(Path(__file__).parent / "water/data/single")]
#         self.config["training"]["training_data"]["systems"] = self.data_file
#         self.config["training"]["validation_data"]["systems"] = self.data_file
#         self.config["model"] = deepcopy(model_se_e2_a)
#         self.config["training"]["numb_steps"] = 1
#         self.config["training"]["save_freq"] = 1
#         self.mixed_types = False
#
#
# class TestEnergyZBLModelSeA(FinetuneTest, unittest.TestCase):
#     def setUp(self):
#         input_json = str(Path(__file__).parent / "water/se_atten.json")
#         with open(input_json) as f:
#             self.config = json.load(f)
#         self.data_file = [str(Path(__file__).parent / "water/data/single")]
#         self.config["training"]["training_data"]["systems"] = self.data_file
#         self.config["training"]["validation_data"]["systems"] = self.data_file
#         self.config["model"] = deepcopy(model_zbl)
#         self.config["training"]["numb_steps"] = 1
#         self.config["training"]["save_freq"] = 1
#         self.mixed_types = False
#


class TestEnergyModelDPA1(FinetuneTest, unittest.TestCase):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = deepcopy(model_dpa1)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.mixed_types = True


#
# class TestEnergyModelDPA2(FinetuneTest, unittest.TestCase):
#     def setUp(self):
#         input_json = str(Path(__file__).parent / "water/se_atten.json")
#         with open(input_json) as f:
#             self.config = json.load(f)
#         self.data_file = [str(Path(__file__).parent / "water/data/single")]
#         self.config["training"]["training_data"]["systems"] = self.data_file
#         self.config["training"]["validation_data"]["systems"] = self.data_file
#         self.config["model"] = deepcopy(model_dpa2)
#         self.config["model"]["descriptor"]["repformer"]["nlayers"] = 2
#
#         self.config["training"]["numb_steps"] = 1
#         self.config["training"]["save_freq"] = 1
#         self.mixed_types = True


if __name__ == "__main__":
    unittest.main()
