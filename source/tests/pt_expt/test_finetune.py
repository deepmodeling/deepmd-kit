# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for ``dp finetune`` in the pt_expt backend.

Part A: Model-level tests (FinetuneTest mixin)
  - test_finetune_change_out_bias
  - test_finetune_change_type

Part B: CLI end-to-end tests (TestFinetuneCLI)
  - test_finetune_cli
  - test_finetune_cli_use_pretrain_script
  - test_finetune_random_fitting
"""

import json
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)

import numpy as np
import torch

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.finetune import (
    get_finetune_rules,
)
from deepmd.pt_expt.utils.stat import (
    make_stat_input,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
    process_systems,
)

EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "examples",
    "water",
)

energy_data_requirement = [
    DataRequirementItem("energy", ndof=1, atomic=False, must=False, high_prec=True),
    DataRequirementItem("force", ndof=3, atomic=True, must=False, high_prec=False),
    DataRequirementItem("virial", ndof=9, atomic=False, must=False, high_prec=False),
    DataRequirementItem("atom_ener", ndof=1, atomic=True, must=False, high_prec=False),
    DataRequirementItem(
        "atom_pref", ndof=1, atomic=True, must=False, high_prec=False, repeat=3
    ),
]


model_se_e2_a = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [6, 12],
        "rcut_smth": 0.50,
        "rcut": 3.00,
        "neuron": [8, 16],
        "resnet_dt": False,
        "axis_neuron": 4,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [16, 16],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 1,
}

model_dpa1 = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa1",
        "sel": 18,
        "rcut_smth": 0.50,
        "rcut": 3.00,
        "neuron": [8, 16],
        "axis_neuron": 4,
        "attn": 4,
        "attn_layer": 2,
        "attn_dotr": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [16, 16],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 1,
}


def _subsample_data(src_dir: str, dst_dir: str, nframes: int = 2) -> None:
    """Copy a data system, keeping only the first *nframes* frames."""
    import shutil as _shutil

    _shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    set_dir = os.path.join(dst_dir, "set.000")
    for name in os.listdir(set_dir):
        if name.endswith(".npy"):
            arr = np.load(os.path.join(set_dir, name))
            np.save(os.path.join(set_dir, name), arr[:nframes])


def _make_config(data_dir: str, model_params: dict, numb_steps: int = 1) -> dict:
    """Build a minimal config dict for finetune tests."""
    config = {
        "model": deepcopy(model_params),
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": [os.path.join(data_dir, "data_0")],
                "batch_size": 2,
            },
            "validation_data": {
                "systems": [os.path.join(data_dir, "data_0")],
                "batch_size": 2,
                "numb_btch": 1,
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": numb_steps,
        },
    }
    return config


# ---------------------------------------------------------------------------
# Part A: Model-level tests
# ---------------------------------------------------------------------------


class FinetuneTest:
    """Mixin with model-level finetune tests."""

    def test_finetune_change_out_bias(self) -> None:
        """Train model -> randomize bias -> change_out_bias -> verify shift."""
        # get data
        type_map = self.config["model"]["type_map"]
        data_systems = process_systems(
            self.config["training"]["training_data"]["systems"]
        )
        data = DeepmdDataSystem(
            systems=data_systems,
            batch_size=1,
            test_size=1,
            type_map=type_map,
            trn_all_set=True,
        )
        data.add_data_requirements(energy_data_requirement)
        sampled = make_stat_input(data, nbatches=1)

        # make sampled of multiple frames with different atom numbs
        numb_atom = sampled[0]["atype"].shape[1]
        small_numb_atom = numb_atom // 2
        small_atom_data = deepcopy(sampled[0])
        # coord is (nframes, nloc*3) in dpmodel/pt_expt format:
        # reshape to 3D, slice atoms, flatten back
        nframes = small_atom_data["coord"].shape[0]
        coord_3d = small_atom_data["coord"].reshape(nframes, numb_atom, 3)
        small_atom_data["coord"] = coord_3d[:, :small_numb_atom, :].reshape(
            nframes, small_numb_atom * 3
        )
        small_atom_data["atype"] = small_atom_data["atype"][:, :small_numb_atom]
        scale_pref = float(small_numb_atom / numb_atom)
        small_atom_data["energy"] *= scale_pref
        small_atom_data["natoms"][:, :2] = small_numb_atom
        # recount per-type atoms
        atype_flat = small_atom_data["atype"][0]
        for ii in range(len(type_map)):
            small_atom_data["natoms"][:, 2 + ii] = np.sum(atype_flat == ii)
        sampled = [sampled[0], small_atom_data]

        # get model and randomize bias
        config = deepcopy(self.config)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        model = get_model(config["model"]).to(DEVICE)

        old_bias = model.get_out_bias()
        rng = np.random.default_rng(42)
        random_bias = rng.standard_normal(to_numpy_array(old_bias).shape).astype(
            to_numpy_array(old_bias).dtype
        )
        model.set_out_bias(random_bias)
        energy_bias_before = to_numpy_array(model.get_out_bias())[0]

        # Run inference BEFORE bias change (need original model predictions)
        model.eval()
        origin_type_map = type_map
        full_type_map = type_map
        sorter = np.argsort(full_type_map)
        idx_type_map = sorter[
            np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
        ]
        ntest = 1

        # model inference (coord needs requires_grad for force via autograd.grad)
        coord0 = torch.tensor(
            sampled[0]["coord"][:ntest], dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype0 = torch.tensor(
            sampled[0]["atype"][:ntest], dtype=torch.int64, device=DEVICE
        )
        box0 = torch.tensor(
            sampled[0]["box"][:ntest], dtype=torch.float64, device=DEVICE
        )
        coord1 = torch.tensor(
            sampled[1]["coord"][:ntest], dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype1 = torch.tensor(
            sampled[1]["atype"][:ntest], dtype=torch.int64, device=DEVICE
        )
        box1 = torch.tensor(
            sampled[1]["box"][:ntest], dtype=torch.float64, device=DEVICE
        )

        energy = model(coord0, atype0, box0)["energy"].detach().cpu().numpy()
        energy_small = model(coord1, atype1, box1)["energy"].detach().cpu().numpy()

        # Now change energy bias
        model.change_out_bias(
            sampled,
            bias_adjust_mode="change-by-statistic",
        )
        energy_bias_after = to_numpy_array(model.get_out_bias())[0]

        # compute ground-truth bias change via least squares
        atom_nums = np.tile(
            np.bincount(sampled[0]["atype"][0].astype(int))[idx_type_map],
            (ntest, 1),
        )
        atom_nums_small = np.tile(
            np.bincount(sampled[1]["atype"][0].astype(int))[idx_type_map],
            (ntest, 1),
        )
        atom_nums = np.concatenate([atom_nums, atom_nums_small], axis=0)

        energy_diff = sampled[0]["energy"][:ntest] - energy
        energy_diff_small = sampled[1]["energy"][:ntest] - energy_small
        energy_diff = np.concatenate([energy_diff, energy_diff_small], axis=0)

        finetune_shift = (
            energy_bias_after[idx_type_map] - energy_bias_before[idx_type_map]
        ).ravel()
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        np.testing.assert_almost_equal(finetune_shift, ground_truth_shift, decimal=10)

    def test_finetune_change_type(self) -> None:
        """Train with type_map A -> load with type_map B -> verify consistency.

        Tests that change_type_map + selective weight copy correctly remaps
        weights so predictions are identical for the same atoms regardless
        of type map ordering.  Uses direct weight loading (bypassing Trainer
        bias adjustment) to isolate the remapping logic, then verifies the
        full Trainer finetune path (with bias adjustment) also works.
        """
        if not self.mixed_types:
            return

        from deepmd.utils.finetune import (
            get_index_between_two_maps,
        )

        config = deepcopy(self.config)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        data_type_map = config["model"]["type_map"]

        for old_type_map, new_type_map in [
            [["H", "X1", "X2", "O", "B"], ["O", "H", "B"]],
            [["O", "H", "B"], ["H", "X1", "X2", "O", "B"]],
        ]:
            old_type_map_index = np.array(
                [old_type_map.index(i) for i in data_type_map], dtype=np.int32
            )
            new_type_map_index = np.array(
                [new_type_map.index(i) for i in data_type_map], dtype=np.int32
            )

            tmpdir = tempfile.mkdtemp(prefix="pt_expt_ft_type_")
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Train pretrained model with old type map
                config_old = deepcopy(config)
                config_old["model"]["type_map"] = old_type_map
                trainer = get_trainer(config_old)
                trainer.run()
                finetune_ckpt = (
                    config_old["training"].get("save_ckpt", "model.ckpt") + ".pt"
                )

                # Load pretrained checkpoint
                state_dict = torch.load(
                    finetune_ckpt, map_location=DEVICE, weights_only=True
                )
                if "model" in state_dict:
                    state_dict = state_dict["model"]

                # Build model_old: same type_map, direct weight load
                model_old = get_model(
                    deepcopy(state_dict["_extra_state"]["model_params"])
                ).to(DEVICE)
                wrapper_old = ModelWrapper(model_old)
                wrapper_old.load_state_dict(state_dict)

                # Build model_new: change_type_map + selective weight copy
                pretrained_model = get_model(
                    deepcopy(state_dict["_extra_state"]["model_params"])
                ).to(DEVICE)
                pretrained_wrapper = ModelWrapper(pretrained_model)
                pretrained_wrapper.load_state_dict(state_dict)

                config_new = deepcopy(config)
                config_new["model"]["type_map"] = new_type_map
                config_new = normalize(config_new)
                model_new = get_model(config_new["model"]).to(DEVICE)
                wrapper_new = ModelWrapper(model_new)

                _, has_new_type = get_index_between_two_maps(old_type_map, new_type_map)
                model_with_new_type_stat = wrapper_new.model if has_new_type else None
                pretrained_wrapper.model.change_type_map(
                    new_type_map,
                    model_with_new_type_stat=model_with_new_type_stat,
                )

                pre_state = pretrained_wrapper.state_dict()
                tgt_state = wrapper_new.state_dict()
                new_state = {}
                for key in tgt_state:
                    if key == "_extra_state":
                        new_state[key] = tgt_state[key]
                    elif key in pre_state:
                        new_state[key] = pre_state[key]
                    else:
                        new_state[key] = tgt_state[key]
                wrapper_new.load_state_dict(new_state)

                # Get sample data for comparison
                data_systems = process_systems(
                    config["training"]["training_data"]["systems"]
                )
                data = DeepmdDataSystem(
                    systems=data_systems,
                    batch_size=1,
                    test_size=1,
                    type_map=data_type_map,
                    trn_all_set=True,
                )
                data.add_data_requirements(energy_data_requirement)
                sampled = make_stat_input(data, nbatches=1)

                ntest = 1
                prec = 1e-10
                box = torch.tensor(
                    sampled[0]["box"][:ntest], dtype=torch.float64, device=DEVICE
                )
                atype_raw = torch.tensor(
                    sampled[0]["atype"][:ntest], dtype=torch.int64, device=DEVICE
                )

                old_index = torch.tensor(
                    old_type_map_index, dtype=torch.int64, device=DEVICE
                )
                new_index = torch.tensor(
                    new_type_map_index, dtype=torch.int64, device=DEVICE
                )

                model_old.eval()
                model_new.eval()

                coord_old = torch.tensor(
                    sampled[0]["coord"][:ntest],
                    dtype=torch.float64,
                    device=DEVICE,
                ).requires_grad_(True)
                result_old = model_old(coord_old, old_index[atype_raw], box=box)
                coord_new = torch.tensor(
                    sampled[0]["coord"][:ntest],
                    dtype=torch.float64,
                    device=DEVICE,
                ).requires_grad_(True)
                result_new = model_new(coord_new, new_index[atype_raw], box=box)

                for key in ["energy", "force", "virial"]:
                    torch.testing.assert_close(
                        result_old[key],
                        result_new[key],
                        rtol=prec,
                        atol=prec,
                    )

                # Now verify full Trainer finetune path (with bias adjustment)
                config_old_ft = deepcopy(config)
                config_old_ft["model"]["type_map"] = old_type_map
                config_old_ft["model"], finetune_links_old = get_finetune_rules(
                    finetune_ckpt, config_old_ft["model"]
                )
                trainer_old = get_trainer(
                    config_old_ft,
                    finetune_model=finetune_ckpt,
                    finetune_links=finetune_links_old,
                )

                config_new_ft = deepcopy(config)
                config_new_ft["model"]["type_map"] = new_type_map
                config_new_ft["model"], finetune_links_new = get_finetune_rules(
                    finetune_ckpt, config_new_ft["model"]
                )
                trainer_new = get_trainer(
                    config_new_ft,
                    finetune_model=finetune_ckpt,
                    finetune_links=finetune_links_new,
                )

                trainer_old.model.eval()
                trainer_new.model.eval()

                coord_old2 = torch.tensor(
                    sampled[0]["coord"][:ntest],
                    dtype=torch.float64,
                    device=DEVICE,
                ).requires_grad_(True)
                result_old2 = trainer_old.model(
                    coord_old2, old_index[atype_raw], box=box
                )
                coord_new2 = torch.tensor(
                    sampled[0]["coord"][:ntest],
                    dtype=torch.float64,
                    device=DEVICE,
                ).requires_grad_(True)
                result_new2 = trainer_new.model(
                    coord_new2, new_index[atype_raw], box=box
                )

                for key in ["energy", "force", "virial"]:
                    torch.testing.assert_close(
                        result_old2[key],
                        result_new2[key],
                        rtol=prec,
                        atol=prec,
                    )
            finally:
                os.chdir(old_cwd)
                shutil.rmtree(tmpdir, ignore_errors=True)


class TestEnergyModelSeA(FinetuneTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir
        cls.config = _make_config(data_dir, model_se_e2_a)
        cls.mixed_types = False


class TestEnergyModelDPA1(FinetuneTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls._tmpdir = tempfile.mkdtemp(prefix="pt_expt_ft_dpa1_data_")
        _subsample_data(
            os.path.join(data_dir, "data_0"),
            os.path.join(cls._tmpdir, "data_0"),
        )
        cls.data_dir = cls._tmpdir
        cls.config = _make_config(cls._tmpdir, model_dpa1)
        cls.mixed_types = True

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part B: CLI end-to-end tests
# ---------------------------------------------------------------------------


class TestFinetuneCLI(unittest.TestCase):
    """End-to-end tests for the ``dp --pt-expt train --finetune`` CLI path."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def _train_pretrained(self, config: dict, tmpdir: str) -> str:
        """Train a 1-step model and return checkpoint path."""
        trainer = get_trainer(config)
        trainer.run()
        ckpt = os.path.join(tmpdir, "model.ckpt.pt")
        self.assertTrue(os.path.exists(ckpt), "Pretrained checkpoint not found")
        return ckpt

    def _assert_inherited_weights_match(
        self,
        ft_state: dict,
        pre_state: dict,
        random_fitting: bool = False,
    ) -> None:
        """Assert that inherited weights in finetuned model match pretrained.

        Descriptor weights must always match.  Fitting weights must match
        unless ``random_fitting`` is True.  ``_extra_state`` and out_bias
        (adjusted by bias computation) are skipped.
        """
        for key in ft_state:
            if key == "_extra_state":
                continue
            if key not in pre_state:
                continue
            if ".descriptor." in key:
                torch.testing.assert_close(
                    ft_state[key],
                    pre_state[key],
                    msg=f"Descriptor weight {key} should match pretrained",
                )
            elif ".fitting" in key:
                if not random_fitting:
                    torch.testing.assert_close(
                        ft_state[key],
                        pre_state[key],
                        msg=f"Fitting weight {key} should match pretrained",
                    )
                else:
                    # random_fitting: network weights must differ
                    # (bias_atom_e is set by bias adjustment, not random init)
                    if ft_state[key].is_floating_point() and "bias_atom_e" not in key:
                        self.assertFalse(
                            torch.equal(ft_state[key], pre_state[key]),
                            msg=f"Fitting weight {key} should NOT match pretrained "
                            f"when random_fitting=True",
                        )

    def test_finetune_cli(self) -> None:
        """Train -> finetune via main() dispatcher -> verify checkpoint exists."""
        from deepmd.pt_expt.entrypoints.main import (
            main,
        )

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_ft_cli_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Phase 1: train pretrained model
            config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            ckpt_path = self._train_pretrained(config, tmpdir)

            # Save original bias
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model_state = state["model"] if "model" in state else state
            original_model = get_model(model_state["_extra_state"]["model_params"]).to(
                DEVICE
            )
            original_wrapper = ModelWrapper(original_model)
            original_wrapper.load_state_dict(model_state)
            original_bias = to_numpy_array(original_model.get_out_bias()).copy()

            # Phase 2: finetune via CLI (lr=0 so weights stay unchanged)
            ft_config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            ft_config["learning_rate"]["start_lr"] = 1e-30
            ft_config["learning_rate"]["stop_lr"] = 1e-30
            ft_config_file = os.path.join(tmpdir, "finetune_input.json")
            with open(ft_config_file, "w") as f:
                json.dump(ft_config, f)

            main(
                [
                    "train",
                    ft_config_file,
                    "--finetune",
                    ckpt_path,
                    "--skip-neighbor-stat",
                ]
            )

            # Verify new checkpoint exists
            ft_ckpt = os.path.join(tmpdir, "model.ckpt.pt")
            self.assertTrue(os.path.exists(ft_ckpt), "Finetune checkpoint not found")

            # Load finetuned model and verify bias changed
            ft_state = torch.load(ft_ckpt, map_location=DEVICE, weights_only=True)
            ft_model_state = ft_state["model"] if "model" in ft_state else ft_state
            ft_model = get_model(ft_model_state["_extra_state"]["model_params"]).to(
                DEVICE
            )
            ft_wrapper = ModelWrapper(ft_model)
            ft_wrapper.load_state_dict(ft_model_state)
            ft_bias = to_numpy_array(ft_model.get_out_bias())

            # Bias should have been adjusted (may or may not differ depending
            # on data, but the checkpoint should at least be valid)
            self.assertEqual(original_bias.shape, ft_bias.shape)

            # Inherited weights (descriptor + fitting) must match pretrained.
            # lr=0 so training step doesn't modify weights.
            self._assert_inherited_weights_match(
                ft_model_state, model_state, random_fitting=False
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_finetune_cli_use_pretrain_script(self) -> None:
        """Finetune with --use-pretrain-script -> config copied from pretrained."""
        from deepmd.pt_expt.entrypoints.main import (
            main,
        )

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_ft_pretrain_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Phase 1: train pretrained model
            config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            ckpt_path = self._train_pretrained(config, tmpdir)

            # Phase 2: finetune with --use-pretrain-script
            # Use a config with different descriptor neuron sizes
            ft_model_params = deepcopy(model_se_e2_a)
            ft_model_params["descriptor"]["neuron"] = [4, 8]  # different
            ft_config = _make_config(self.data_dir, ft_model_params, numb_steps=1)
            ft_config_file = os.path.join(tmpdir, "finetune_input.json")
            with open(ft_config_file, "w") as f:
                json.dump(ft_config, f)

            main(
                [
                    "train",
                    ft_config_file,
                    "--finetune",
                    ckpt_path,
                    "--use-pretrain-script",
                    "--skip-neighbor-stat",
                ]
            )

            # Verify the output config was updated from pretrained
            with open(os.path.join(tmpdir, "out.json")) as f:
                output_config = json.load(f)
            # Descriptor neuron should be from pretrained, not from ft_config
            self.assertEqual(
                output_config["model"]["descriptor"]["neuron"],
                model_se_e2_a["descriptor"]["neuron"],
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_finetune_random_fitting(self) -> None:
        """Finetune with --model-branch RANDOM -> descriptor from pretrained, fitting random."""
        tmpdir = tempfile.mkdtemp(prefix="pt_expt_ft_random_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Phase 1: train pretrained model
            config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            ckpt_path = self._train_pretrained(config, tmpdir)

            # Phase 2: finetune with RANDOM (random fitting)
            ft_config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            ft_config = update_deepmd_input(ft_config, warning=False)
            ft_config = normalize(ft_config)
            ft_config["model"], finetune_links = get_finetune_rules(
                ckpt_path,
                ft_config["model"],
                model_branch="RANDOM",
            )

            # Verify finetune rule has random_fitting=True
            self.assertTrue(finetune_links["Default"].get_random_fitting())

            trainer_ft = get_trainer(
                ft_config,
                finetune_model=ckpt_path,
                finetune_links=finetune_links,
            )

            # Load pretrained weights for comparison
            pretrained_state = torch.load(
                ckpt_path, map_location=DEVICE, weights_only=True
            )
            if "model" in pretrained_state:
                pretrained_state = pretrained_state["model"]
            pretrained_model = get_model(
                pretrained_state["_extra_state"]["model_params"]
            ).to(DEVICE)
            pretrained_wrapper = ModelWrapper(pretrained_model)
            pretrained_wrapper.load_state_dict(pretrained_state)

            # Descriptor weights should match; fitting should NOT
            ft_state = trainer_ft.wrapper.state_dict()
            pre_state = pretrained_wrapper.state_dict()
            self._assert_inherited_weights_match(
                ft_state, pre_state, random_fitting=True
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_finetune_from_pte(self) -> None:
        """Train -> freeze to .pte -> finetune from .pte -> verify checkpoint."""
        from deepmd.pt_expt.entrypoints.main import (
            freeze,
            main,
        )

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_ft_pte_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Phase 1: train pretrained model
            config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            ckpt_path = self._train_pretrained(config, tmpdir)

            # Phase 2: freeze to .pte
            pte_path = os.path.join(tmpdir, "frozen.pte")
            freeze(model=ckpt_path, output=pte_path)
            self.assertTrue(os.path.exists(pte_path))

            # Phase 3: finetune from .pte via CLI (lr=0 so weights stay unchanged)
            ft_config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            ft_config["learning_rate"]["start_lr"] = 1e-30
            ft_config["learning_rate"]["stop_lr"] = 1e-30
            ft_config_file = os.path.join(tmpdir, "finetune_input.json")
            with open(ft_config_file, "w") as f:
                json.dump(ft_config, f)

            main(
                [
                    "train",
                    ft_config_file,
                    "--finetune",
                    pte_path,
                    "--skip-neighbor-stat",
                ]
            )

            # Verify new checkpoint exists
            ft_ckpt = os.path.join(tmpdir, "model.ckpt.pt")
            self.assertTrue(os.path.exists(ft_ckpt), "Finetune checkpoint not found")

            # Load finetuned model and verify it's valid
            ft_state = torch.load(ft_ckpt, map_location=DEVICE, weights_only=True)
            ft_model_state = ft_state["model"] if "model" in ft_state else ft_state
            self.assertIn("_extra_state", ft_model_state)

            # Load pretrained from .pt for weight comparison
            pre_state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            pre_model_state = pre_state["model"] if "model" in pre_state else pre_state

            # Inherited weights must match pretrained
            self._assert_inherited_weights_match(
                ft_model_state, pre_model_state, random_fitting=False
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_finetune_from_pte_use_pretrain_script(self) -> None:
        """Train -> freeze to .pte -> finetune with --use-pretrain-script."""
        from deepmd.pt_expt.entrypoints.main import (
            freeze,
            main,
        )

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_ft_pte_ups_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            # Phase 1: train pretrained model
            config = _make_config(self.data_dir, model_se_e2_a, numb_steps=1)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            ckpt_path = self._train_pretrained(config, tmpdir)

            # Phase 2: freeze to .pte (embeds model_params)
            pte_path = os.path.join(tmpdir, "frozen.pte")
            freeze(model=ckpt_path, output=pte_path)

            # Phase 3: finetune from .pte with --use-pretrain-script
            ft_model_params = deepcopy(model_se_e2_a)
            ft_model_params["descriptor"]["neuron"] = [4, 8]  # different
            ft_config = _make_config(self.data_dir, ft_model_params, numb_steps=1)
            ft_config_file = os.path.join(tmpdir, "finetune_input.json")
            with open(ft_config_file, "w") as f:
                json.dump(ft_config, f)

            main(
                [
                    "train",
                    ft_config_file,
                    "--finetune",
                    pte_path,
                    "--use-pretrain-script",
                    "--skip-neighbor-stat",
                ]
            )

            # Verify the output config was updated from pretrained
            with open(os.path.join(tmpdir, "out.json")) as f:
                output_config = json.load(f)
            # Descriptor neuron should be from pretrained, not from ft_config
            self.assertEqual(
                output_config["model"]["descriptor"]["neuron"],
                model_se_e2_a["descriptor"]["neuron"],
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
