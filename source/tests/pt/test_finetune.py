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
    model_dos,
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
    model_zbl,
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
        "dos",
        ndof=250,
        atomic=False,
        must=False,
        high_prec=True,
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
    def test_finetune_change_out_bias(self) -> None:
        self.testkey = "energy" if self.testkey is None else self.testkey
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
        # make sampled of multiple frames with different atom numbs
        numb_atom = sampled[0]["atype"].shape[1]
        small_numb_atom = numb_atom // 2
        small_atom_data = deepcopy(sampled[0])
        atomic_key = ["coord", "atype"]
        for kk in atomic_key:
            small_atom_data[kk] = small_atom_data[kk][:, :small_numb_atom]
        scale_pref = float(small_numb_atom / numb_atom)
        small_atom_data[self.testkey] *= scale_pref
        small_atom_data["natoms"][:, :2] = small_numb_atom
        small_atom_data["natoms"][:, 2:] = torch.bincount(
            small_atom_data["atype"][0],
            minlength=small_atom_data["natoms"].shape[1] - 2,
        )
        sampled = [sampled[0], small_atom_data]

        # get model
        model = get_model(self.config["model"]).to(env.DEVICE)
        atomic_model = model.atomic_model
        atomic_model["out_bias"] = torch.rand_like(atomic_model["out_bias"])
        energy_bias_before = to_numpy_array(atomic_model["out_bias"])[0]

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
        energy_bias_after = to_numpy_array(atomic_model["out_bias"])[0]

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
        atom_nums_small = np.tile(
            np.bincount(to_numpy_array(sampled[1]["atype"][0]))[idx_type_map],
            (ntest, 1),
        )
        atom_nums = np.concatenate([atom_nums, atom_nums_small], axis=0)

        energy = dp.eval(
            to_numpy_array(sampled[0]["coord"][:ntest]),
            to_numpy_array(sampled[0]["box"][:ntest]),
            to_numpy_array(sampled[0]["atype"][0]),
        )[0]
        energy_small = dp.eval(
            to_numpy_array(sampled[1]["coord"][:ntest]),
            to_numpy_array(sampled[1]["box"][:ntest]),
            to_numpy_array(sampled[1]["atype"][0]),
        )[0]
        energy_diff = to_numpy_array(sampled[0][self.testkey][:ntest]) - energy
        energy_diff_small = (
            to_numpy_array(sampled[1][self.testkey][:ntest]) - energy_small
        )
        energy_diff = np.concatenate([energy_diff, energy_diff_small], axis=0)
        finetune_shift = (
            energy_bias_after[idx_type_map] - energy_bias_before[idx_type_map]
        ).ravel()
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        # check values
        np.testing.assert_almost_equal(finetune_shift, ground_truth_shift, decimal=10)

        self.tearDown()

    def test_finetune_change_type(self) -> None:
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
        for [old_type_map, new_type_map] in [
            [["H", "X1", "X2", "O", "B"], ["O", "H", "B"]],
            [["O", "H", "B"], ["H", "X1", "X2", "O", "B"]],
        ]:
            old_type_map_index = np.array(
                [old_type_map.index(i) for i in data_type_map], dtype=np.int32
            )
            new_type_map_index = np.array(
                [new_type_map.index(i) for i in data_type_map], dtype=np.int32
            )

            # get pretrained model with old type map
            config_old_type_map = deepcopy(self.config)
            config_old_type_map["model"]["type_map"] = old_type_map
            trainer = get_trainer(config_old_type_map)
            trainer.run()
            finetune_model = (
                config_old_type_map["training"].get("save_ckpt", "model.ckpt") + ".pt"
            )

            # finetune load the same type_map
            config_old_type_map_finetune = deepcopy(self.config)
            config_old_type_map_finetune["model"]["type_map"] = old_type_map
            config_old_type_map_finetune["model"], finetune_links = get_finetune_rules(
                finetune_model,
                config_old_type_map_finetune["model"],
            )
            trainer_finetune_old = get_trainer(
                config_old_type_map_finetune,
                finetune_model=finetune_model,
                finetune_links=finetune_links,
            )

            # finetune load the slim type_map
            config_new_type_map_finetune = deepcopy(self.config)
            config_new_type_map_finetune["model"]["type_map"] = new_type_map
            config_new_type_map_finetune["model"], finetune_links = get_finetune_rules(
                finetune_model,
                config_new_type_map_finetune["model"],
            )
            trainer_finetune_new = get_trainer(
                config_new_type_map_finetune,
                finetune_model=finetune_model,
                finetune_links=finetune_links,
            )

            # test consistency
            ntest = 1
            prec = 1e-10
            model_old_result = trainer_finetune_old.model(
                sampled[0]["coord"][:ntest],
                to_torch_tensor(old_type_map_index)[sampled[0]["atype"][:ntest]],
                box=sampled[0]["box"][:ntest],
            )
            model_new_result = trainer_finetune_new.model(
                sampled[0]["coord"][:ntest],
                to_torch_tensor(new_type_map_index)[sampled[0]["atype"][:ntest]],
                box=sampled[0]["box"][:ntest],
            )
            test_keys = ["energy", "force", "virial"]
            for key in test_keys:
                torch.testing.assert_close(
                    model_old_result[key],
                    model_new_result[key],
                    rtol=prec,
                    atol=prec,
                )

            self.tearDown()

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestEnergyModelSeA(FinetuneTest, unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.mixed_types = False
        self.testkey = None


class TestEnergyZBLModelSeA(FinetuneTest, unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = deepcopy(model_zbl)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.mixed_types = False
        self.testkey = None


class TestEnergyDOSModelSeA(FinetuneTest, unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "dos/input.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "dos/data/global_system")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = deepcopy(model_dos)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.mixed_types = False
        self.testkey = "dos"


class TestEnergyModelDPA1(FinetuneTest, unittest.TestCase):
    def setUp(self) -> None:
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
        self.testkey = None


class TestEnergyModelDPA2(FinetuneTest, unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = deepcopy(model_dpa2)
        self.config["model"]["descriptor"]["repformer"]["nlayers"] = 2

        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.mixed_types = True
        self.testkey = None


if __name__ == "__main__":
    unittest.main()
