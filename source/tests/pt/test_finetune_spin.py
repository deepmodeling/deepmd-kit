# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for spin model finetune: change_out_bias, change_type_map, and e2e finetune."""

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
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .model.test_permutation import (
    model_spin,
)

spin_data_requirement = [
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
        "force_mag",
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
        "spin",
        ndof=3,
        atomic=True,
        must=True,
        high_prec=False,
    ),
]


class SpinFinetuneTest:
    """Mixin test class for spin model finetune operations."""

    def test_change_out_bias(self) -> None:
        """Test that change_out_bias correctly adjusts energy bias for spin model."""
        # get data
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(spin_data_requirement)
        sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )

        # get model
        model = get_model(self.config["model"]).to(env.DEVICE)

        # set random bias
        atomic_model = model.backbone_model.atomic_model
        atomic_model["out_bias"] = torch.rand_like(atomic_model["out_bias"])
        energy_bias_before = to_numpy_array(atomic_model["out_bias"])[0]

        # prepare original model for prediction
        dp = torch.jit.script(model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(dp, tmp_model.name)
        dp = DeepEval(tmp_model.name)

        origin_type_map = self.config["model"]["type_map"][:2]
        full_type_map = self.config["model"]["type_map"]

        # change energy bias via spin model's change_out_bias
        model.change_out_bias(
            sampled,
            bias_adjust_mode="change-by-statistic",
        )
        energy_bias_after = to_numpy_array(atomic_model["out_bias"])[0]

        # get ground-truth energy bias change via least squares
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
            spin=to_numpy_array(sampled[0]["spin"][:ntest]),
        )[0]

        energy_diff = to_numpy_array(sampled[0]["energy"][:ntest]) - energy
        finetune_shift = (
            energy_bias_after[idx_type_map] - energy_bias_before[idx_type_map]
        ).ravel()
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        # check values
        np.testing.assert_almost_equal(finetune_shift, ground_truth_shift, decimal=10)
        os.unlink(tmp_model.name)

    def test__get_spin_sampled_func(self) -> None:
        """Test that _get_spin_sampled_func correctly transforms spin data."""
        # get data
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(spin_data_requirement)
        sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )

        model = get_model(self.config["model"]).to(env.DEVICE)

        # Create a sampled_func callable
        def sampled_func():
            return sampled

        spin_sampled_func = model._get_spin_sampled_func(sampled_func)
        spin_sampled = spin_sampled_func()

        # Verify the transformed data
        for i, sys_data in enumerate(spin_sampled):
            original = sampled[i]
            nloc = original["atype"].shape[1]
            # coord should be doubled (real + virtual)
            assert sys_data["coord"].shape[1] == 2 * nloc
            # atype should be doubled
            assert sys_data["atype"].shape[1] == 2 * nloc
            # spin should not be in the transformed data
            assert "spin" not in sys_data
            # energy should be preserved
            if "energy" in original:
                torch.testing.assert_close(sys_data["energy"], original["energy"])
            # natoms should be transformed correctly
            if "natoms" in original:
                natoms = original["natoms"]
                expected_natoms = torch.cat(
                    [2 * natoms[:, :2], natoms[:, 2:], natoms[:, 2:]], dim=-1
                )
                torch.testing.assert_close(sys_data["natoms"], expected_natoms)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestSpinFinetuneSeA(SpinFinetuneTest, unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "NiO/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = deepcopy(model_spin)
        self.config["model"]["type_map"] = ["Ni", "O"]
        self.config["model"]["descriptor"] = {
            "type": "se_e2_a",
            "sel": [20, 20],
            "rcut_smth": 0.50,
            "rcut": 4.00,
            "neuron": [25, 50, 100],
            "resnet_dt": False,
            "axis_neuron": 16,
            "seed": 1,
        }
        self.config["model"]["fitting_net"] = {
            "neuron": [24, 24, 24],
            "resnet_dt": True,
            "seed": 1,
        }
        self.config["model"]["spin"] = {
            "use_spin": [True, False],
            "virtual_scale": [0.3140],
        }
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["loss"] = {
            "type": "ener_spin",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_fr": 1000,
            "limit_pref_fr": 1,
            "start_pref_fm": 1000,
            "limit_pref_fm": 1,
        }
        self.mixed_types = False


class SpinFinetuneE2ETest:
    """End-to-end test mixin for spin model finetune workflow.

    Tests the full workflow: train from scratch -> save -> finetune with change_out_bias.
    """

    def test_finetune_e2e(self) -> None:
        """Test the full finetune workflow for a spin model."""
        # Step 1: Train from scratch
        config_pretrain = deepcopy(self.config)
        config_pretrain["training"]["save_ckpt"] = "model"
        trainer = get_trainer(config_pretrain)
        trainer.run()
        finetune_model = (
            config_pretrain["training"].get("save_ckpt", "model.ckpt") + ".pt"
        )

        # Step 2: Finetune with the same type_map (should work after the fix)
        config_finetune = deepcopy(self.config)
        config_finetune["model"], finetune_links = get_finetune_rules(
            finetune_model,
            config_finetune["model"],
        )
        # This should NOT raise an error after the fix
        trainer_finetune = get_trainer(
            config_finetune,
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # Verify the model is functional after finetune loading
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(spin_data_requirement)
        sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )
        # Run inference to verify model works
        ntest = 1
        result = trainer_finetune.model(
            sampled[0]["coord"][:ntest],
            sampled[0]["atype"][:ntest],
            spin=sampled[0]["spin"][:ntest],
            box=sampled[0]["box"][:ntest],
        )
        # Basic checks - model should produce valid outputs
        assert "energy" in result
        assert "force" in result
        assert result["energy"].shape == (ntest, 1)
        nloc = sampled[0]["atype"].shape[1]
        assert result["force"].shape == (ntest, nloc, 3)

    def test_finetune_change_type_map(self) -> None:
        """Test change_type_map for spin model.

        Only runs for mixed_types descriptors since se_e2_a
        does not support type map extension.
        """
        if not self.mixed_types:
            return
        # Train a pretrained model
        config_pretrain = deepcopy(self.config)
        config_pretrain["training"]["save_ckpt"] = "model"
        trainer = get_trainer(config_pretrain)
        trainer.run()
        finetune_model = (
            config_pretrain["training"].get("save_ckpt", "model.ckpt") + ".pt"
        )

        # Finetune with a new type_map that has extra types
        config_finetune = deepcopy(self.config)
        config_finetune["model"]["type_map"] = self.config["model"]["type_map"] + ["Fe"]
        # Extend spin config for the new type
        config_finetune["model"]["spin"]["use_spin"] = self.config["model"]["spin"][
            "use_spin"
        ] + [False]
        config_finetune["model"], finetune_links = get_finetune_rules(
            finetune_model,
            config_finetune["model"],
        )
        # This should NOT raise an error: the spin model should handle type_map change
        trainer_finetune = get_trainer(
            config_finetune,
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # Verify the new type map is applied correctly
        new_type_map = trainer_finetune.model.get_type_map()
        assert "Fe" in new_type_map

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "checkpoint"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestSpinFinetuneE2ESeA(SpinFinetuneE2ETest, unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "NiO/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = {
            "type_map": ["Ni", "O"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
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
            "spin": {
                "use_spin": [True, False],
                "virtual_scale": [0.3140],
            },
        }
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["loss"] = {
            "type": "ener_spin",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_fr": 1000,
            "limit_pref_fr": 1,
            "start_pref_fm": 1000,
            "limit_pref_fm": 1,
        }
        self.mixed_types = False


class TestSpinFinetuneWithDefaultFparam(unittest.TestCase):
    """Test spin model finetune with default_fparam enabled.

    Verifies that _make_wrapped_sampler correctly injects default fparam
    into sampled data when spin preprocessing is also active.
    """

    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.data_file = [str(Path(__file__).parent / "NiO/data/single")]
        self.config["training"]["training_data"]["systems"] = self.data_file
        self.config["training"]["validation_data"]["systems"] = self.data_file
        self.config["model"] = {
            "type_map": ["Ni", "O"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
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
                "numb_fparam": 2,
                "default_fparam": [0.5, 1.0],
            },
            "spin": {
                "use_spin": [True, False],
                "virtual_scale": [0.3140],
            },
        }
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["loss"] = {
            "type": "ener_spin",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_fr": 1000,
            "limit_pref_fr": 1,
            "start_pref_fm": 1000,
            "limit_pref_fm": 1,
        }

    def test_spin_sampled_func_with_default_fparam(self) -> None:
        """Test that _get_spin_sampled_func + _make_wrapped_sampler injects fparam."""
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(spin_data_requirement)
        sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )

        model = get_model(self.config["model"]).to(env.DEVICE)

        # Verify model has default_fparam
        assert model.backbone_model.atomic_model.has_default_fparam()

        # sampled should NOT have fparam yet
        assert "fparam" not in sampled[0]

        def sampled_func():
            return sampled

        # _get_spin_sampled_func chains: spin preprocess -> _make_wrapped_sampler
        spin_sampled_func = model._get_spin_sampled_func(sampled_func)
        spin_sampled = spin_sampled_func()

        for sys_data in spin_sampled:
            # fparam should be injected by _make_wrapped_sampler
            assert "fparam" in sys_data, (
                "_make_wrapped_sampler did not inject default fparam"
            )
            nframe = sys_data["atype"].shape[0]
            assert sys_data["fparam"].shape == (nframe, 2)
            # check values match default_fparam
            np.testing.assert_allclose(
                to_numpy_array(sys_data["fparam"][0]),
                [0.5, 1.0],
            )

    def test_finetune_e2e_with_default_fparam(self) -> None:
        """Test e2e finetune for spin model with default_fparam."""
        config_pretrain = deepcopy(self.config)
        config_pretrain["training"]["save_ckpt"] = "model"
        trainer = get_trainer(config_pretrain)
        trainer.run()
        finetune_model = (
            config_pretrain["training"].get("save_ckpt", "model.ckpt") + ".pt"
        )

        config_finetune = deepcopy(self.config)
        config_finetune["model"], finetune_links = get_finetune_rules(
            finetune_model,
            config_finetune["model"],
        )
        # Should not raise an error with spin + default_fparam
        trainer_finetune = get_trainer(
            config_finetune,
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # Verify the model works
        data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.config["model"]["type_map"],
        )
        data.add_data_requirement(spin_data_requirement)
        sampled = make_stat_input(
            data.systems,
            data.dataloaders,
            nbatches=1,
        )
        ntest = 1
        result = trainer_finetune.model(
            sampled[0]["coord"][:ntest],
            sampled[0]["atype"][:ntest],
            spin=sampled[0]["spin"][:ntest],
            box=sampled[0]["box"][:ntest],
        )
        assert "energy" in result
        assert result["energy"].shape == (ntest, 1)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "checkpoint"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


if __name__ == "__main__":
    unittest.main()
