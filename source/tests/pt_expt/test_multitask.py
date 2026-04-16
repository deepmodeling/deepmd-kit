# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multi-task training in the pt_expt backend.

Verifies that:
1. Multi-task training completes without error for various descriptors
2. Shared descriptor parameters are identical between tasks
3. lcurve.out has per-model columns
4. Checkpoint save/load roundtrip works
5. Multi-task freeze extracts single head correctly
6. Shared fitting_net with case_embd works (share_fitting)
7. Shared fitting stat (fparam_avg/fparam_inv_std) are shared between models
8. Case embedding with 3 models and dim_case_embd=3 works correctly
9. Multi-task descriptor gradients match sum of single-task gradients
"""

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
from unittest.mock import (
    patch,
)

import numpy as np
import torch

import deepmd.utils.random as dp_random
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
from deepmd.pt_expt.utils.multi_task import (
    preprocess_shared_params,
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

_energy_data_requirement = [
    DataRequirementItem("energy", ndof=1, atomic=False, must=False, high_prec=True),
    DataRequirementItem("force", ndof=3, atomic=True, must=False, high_prec=False),
    DataRequirementItem("virial", ndof=9, atomic=False, must=False, high_prec=False),
]

# Paths to the water data used by PT tests
_PT_DATA = str(Path(__file__).parent.parent / "pt" / "water" / "data" / "data_0")


def _skip_if_no_data() -> None:
    if not os.path.isdir(_PT_DATA):
        raise unittest.SkipTest(f"Test data not found: {_PT_DATA}")


# ---------------------------------------------------------------------------
# Descriptor configs (small models for fast testing)
# ---------------------------------------------------------------------------
_descriptor_se_e2_a = {
    "type": "se_e2_a",
    "sel": [6, 12],
    "rcut_smth": 0.50,
    "rcut": 3.00,
    "neuron": [8, 16],
    "resnet_dt": False,
    "axis_neuron": 4,
    "type_one_side": True,
    "seed": 1,
}

_descriptor_dpa1 = {
    "type": "se_atten",
    "sel": 18,
    "rcut_smth": 0.5,
    "rcut": 3.0,
    "neuron": [8, 16],
    "axis_neuron": 4,
    "attn": 16,
    "attn_layer": 2,
    "attn_dotr": True,
    "attn_mask": False,
    "set_davg_zero": True,
    "type_one_side": True,
    "seed": 1,
}

_descriptor_dpa2 = {
    "type": "dpa2",
    "repinit": {
        "rcut": 4.0,
        "rcut_smth": 0.5,
        "nsel": 18,
        "neuron": [2, 4, 8],
        "axis_neuron": 4,
        "activation_function": "tanh",
    },
    "repformer": {
        "rcut": 3.0,
        "rcut_smth": 0.5,
        "nsel": 12,
        "nlayers": 2,
        "g1_dim": 8,
        "g2_dim": 5,
        "attn2_hidden": 3,
        "attn2_nhead": 1,
        "attn1_hidden": 5,
        "attn1_nhead": 1,
        "axis_neuron": 4,
        "update_h2": False,
        "update_g1_has_conv": True,
        "update_g1_has_grrg": True,
        "update_g1_has_drrd": True,
        "update_g1_has_attn": True,
        "update_g2_has_g1g1": True,
        "update_g2_has_attn": True,
        "attn2_has_gate": True,
    },
    "seed": 1,
    "add_tebd_to_repinit_out": False,
}

_descriptor_dpa3 = {
    "type": "dpa3",
    "repflow": {
        "n_dim": 8,
        "e_dim": 5,
        "a_dim": 4,
        "nlayers": 2,
        "e_rcut": 3.0,
        "e_rcut_smth": 0.5,
        "e_sel": 12,
        "a_rcut": 3.0,
        "a_rcut_smth": 0.5,
        "a_sel": 8,
        "axis_neuron": 4,
        "a_compress_rate": 1,
        "a_compress_e_rate": 2,
        "a_compress_use_split": True,
        "update_angle": True,
        "update_style": "res_residual",
        "update_residual": 0.1,
        "update_residual_init": "const",
        "smooth_edge_update": True,
    },
    "activation_function": "silut:10.0",
    "use_tebd_bias": False,
    "precision": "float32",
    "concat_output_tebd": False,
}

_fitting_net = {
    "neuron": [16, 16],
    "resnet_dt": True,
    "seed": 1,
}


def _make_multitask_config(
    descriptor: dict,
    data_dir: str = _PT_DATA,
    numb_steps: int = 1,
    share_fitting: bool = False,
) -> dict:
    """Build a multi-task config with the given descriptor."""
    type_map = ["O", "H"]
    fitting = deepcopy(_fitting_net)

    shared_dict: dict = {
        "my_type_map": type_map,
        "my_descriptor": deepcopy(descriptor),
    }

    if share_fitting:
        shared_fitting = deepcopy(fitting)
        shared_fitting["dim_case_embd"] = 2
        shared_dict["my_fitting"] = shared_fitting
        fitting_ref_1: dict | str = "my_fitting"
        fitting_ref_2: dict | str = "my_fitting"
    else:
        fitting_ref_1 = deepcopy(fitting)
        fitting_ref_2 = deepcopy(fitting)

    config = {
        "model": {
            "shared_dict": shared_dict,
            "model_dict": {
                "model_1": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": fitting_ref_1,
                    "data_stat_nbatch": 1,
                },
                "model_2": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": fitting_ref_2,
                    "data_stat_nbatch": 1,
                },
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss_dict": {
            "model_1": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
            "model_2": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
        },
        "training": {
            "model_prob": {
                "model_1": 0.5,
                "model_2": 0.5,
            },
            "data_dict": {
                "model_1": {
                    "stat_file": "./stat_files/model_1",
                    "training_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
                "model_2": {
                    "stat_file": "./stat_files/model_2",
                    "training_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": numb_steps,
        },
    }
    return config


class MultiTaskTrainTest:
    """Mixin that tests multi-task training for a particular descriptor type.

    Subclasses must set ``self.config``, ``self.shared_links``,
    and ``self.share_fitting`` before calling these test methods.
    """

    def test_multitask_train(self) -> None:
        """Train, verify lcurve format and shared params."""
        trainer = get_trainer(deepcopy(self.config), shared_links=self.shared_links)
        trainer.run()

        # --- lcurve.out format ---
        lcurve_path = "lcurve.out"
        self.assertTrue(os.path.exists(lcurve_path), "lcurve.out not created")
        with open(lcurve_path) as f:
            lines = f.readlines()
        header_line = lines[0]
        header_cols = header_line.strip().lstrip("#").split()
        model_keys = list(self.config["training"]["model_prob"].keys())
        for mk in model_keys:
            cols_for_model = [c for c in header_cols if mk in c]
            self.assertGreater(
                len(cols_for_model), 0, f"No lcurve columns found for {mk}"
            )
        data_lines = [line for line in lines if not line.startswith("#")]
        self.assertGreater(len(data_lines), 0, "No data lines in lcurve.out")
        data_cols = data_lines[0].split()
        self.assertEqual(len(data_cols), len(header_cols))

        # --- model keys ---
        self.assertEqual(len(trainer.wrapper.model), 2)
        self.assertIn("model_1", trainer.wrapper.model)
        self.assertIn("model_2", trainer.wrapper.model)

        # --- shared descriptor params are identical ---
        multi_state_dict = trainer.wrapper.model.state_dict()
        for state_key in multi_state_dict:
            if "model_1" in state_key:
                partner_key = state_key.replace("model_1", "model_2")
                self.assertIn(partner_key, multi_state_dict)
            if "model_2" in state_key:
                partner_key = state_key.replace("model_2", "model_1")
                self.assertIn(partner_key, multi_state_dict)

            is_descriptor = "model_1.atomic_model.descriptor" in state_key
            is_shared_fitting = (
                self.share_fitting
                and "model_1.atomic_model.fitting_net" in state_key
                and "fitting_net.bias_atom_e" not in state_key
                and "fitting_net.case_embd" not in state_key
            )
            if is_descriptor or is_shared_fitting:
                partner_key = state_key.replace("model_1", "model_2")
                torch.testing.assert_close(
                    multi_state_dict[state_key],
                    multi_state_dict[partner_key],
                    msg=f"Shared param mismatch: {state_key}",
                )

        # --- checkpoint exists ---
        ckpt_files = [f for f in os.listdir(".") if f.endswith(".pt")]
        self.assertGreater(len(ckpt_files), 0, "No checkpoint files saved")

        # --- case_embd verification (share_fitting only) ---
        # Verify that each branch's case_embd is a distinct one-hot vector
        # matching the alphabetical sort order, so the shared fitting net
        # can distinguish which training dataset is being used.
        if self.share_fitting:
            ce1 = trainer.wrapper.model["model_1"].atomic_model.fitting_net.case_embd
            ce2 = trainer.wrapper.model["model_2"].atomic_model.fitting_net.case_embd
            self.assertIsNotNone(ce1, "case_embd not set on model_1")
            self.assertIsNotNone(ce2, "case_embd not set on model_2")
            dim = ce1.shape[0]
            # Sorted keys: ["model_1", "model_2"] → indices 0, 1
            expected_eye = torch.eye(dim, dtype=ce1.dtype, device=ce1.device)
            torch.testing.assert_close(
                ce1,
                expected_eye[0],
                msg="model_1 case_embd should be one-hot index 0 (alphabetical order)",
            )
            torch.testing.assert_close(
                ce2,
                expected_eye[1],
                msg="model_2 case_embd should be one-hot index 1 (alphabetical order)",
            )
            # case_embd should NOT be shared in state_dict
            for state_key in multi_state_dict:
                if (
                    "model_1.atomic_model.fitting_net" in state_key
                    and "case_embd" in state_key
                ):
                    partner_key = state_key.replace("model_1", "model_2")
                    self.assertFalse(
                        torch.equal(
                            multi_state_dict[state_key],
                            multi_state_dict[partner_key],
                        ),
                        f"case_embd should NOT be shared: {state_key}",
                    )

    def test_multitask_finetune(self) -> None:
        """Train, then finetune with 4 branches from pretrained 2-branch model.

        For mixed_types descriptors, uses extended type_map ["O","H","B"] to test
        change_type_map + model_with_new_type_stat integration.  For non-mixed_types
        descriptors, uses same type_map ["O","H"].

        Builds a reference state_dict by manually replicating the trainer's
        finetune operations (load pretrained, change_type_map, weight copy) and
        verifies per-branch weight inheritance:
          - model_1: resume (ALL weights match reference)
          - model_2: finetune from model_2 (all except out_bias/out_std match)
          - model_3: finetune from model_2 as new head (cross-branch key remap)
          - model_4: random fitting (descriptor from pretrained, random fitting_net)
        """
        from deepmd.pt_expt.utils.finetune import (
            get_finetune_rules,
        )

        # Phase 1: train pretrained 2-branch model (2 steps)
        config_pretrain = _make_multitask_config(
            self.descriptor, share_fitting=self.share_fitting, numb_steps=2
        )
        config_pretrain["training"]["save_freq"] = 2
        config_pretrain["model"], shared_links_pre = preprocess_shared_params(
            config_pretrain["model"]
        )
        config_pretrain = update_deepmd_input(config_pretrain, warning=False)
        config_pretrain = normalize(config_pretrain, multi_task=True)
        trainer = get_trainer(config_pretrain, shared_links=shared_links_pre)
        trainer.run()

        ckpt_path = os.path.join(os.getcwd(), "model.ckpt.pt")
        self.assertTrue(os.path.exists(ckpt_path), "Pretrained checkpoint not created")

        # Phase 2: build reference state_dict
        # For mixed_types: extend type_map to ["O","H","B"], build
        # model_with_new_type_stat with computed stats, and apply
        # change_type_map on pretrained.
        # For non-mixed_types: use pretrained state directly (no extension).
        ft_type_map = ["O", "H", "B"] if self.mixed_types else ["O", "H"]

        state_dict_full = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        state_dict_ckpt = (
            state_dict_full["model"] if "model" in state_dict_full else state_dict_full
        )
        pretrained_model_params = state_dict_ckpt["_extra_state"]["model_params"]

        # Build pretrained wrapper (separate model per branch)
        pretrained_models = {}
        for pk in pretrained_model_params["model_dict"]:
            pretrained_models[pk] = get_model(
                deepcopy(pretrained_model_params["model_dict"][pk])
            ).to(DEVICE)
        pretrained_wrapper = ModelWrapper(pretrained_models)
        pretrained_wrapper.load_state_dict(state_dict_ckpt)

        # Record pretrained state BEFORE change_type_map — used later to
        # verify O/H stats are inherited from pretrained, not recomputed.
        pretrained_oh_state = {
            k: v.clone() for k, v in pretrained_wrapper.model.state_dict().items()
        }

        if self.mixed_types:
            # Build a model with extended type_map and compute stats so that
            # the new type ("B", unseen in data) gets proper default stats
            # (davg=0, dstd=0.1) instead of the no-stat defaults (0/1).
            ref_model_params = deepcopy(
                pretrained_model_params["model_dict"]["model_1"]
            )
            ref_model_params["type_map"] = ft_type_map
            ref_model = get_model(ref_model_params).to(DEVICE)

            data_systems = process_systems([_PT_DATA])
            data = DeepmdDataSystem(
                systems=data_systems,
                batch_size=1,
                test_size=1,
                type_map=ft_type_map,
                trn_all_set=True,
            )
            data.add_data_requirements(_energy_data_requirement)
            ref_model.compute_or_load_stat(
                sampled_func=lambda: make_stat_input(data, 1),
                stat_file_path=None,
            )

            # Apply change_type_map on each pretrained branch
            for pk in pretrained_model_params["model_dict"]:
                pretrained_wrapper.model[pk].change_type_map(
                    ft_type_map,
                    model_with_new_type_stat=ref_model,
                )

        ref_state_dict = pretrained_wrapper.model.state_dict()

        # Phase 3: build 4-branch finetune config
        finetune_config = _make_multitask_config(
            self.descriptor, share_fitting=self.share_fitting
        )
        if self.mixed_types:
            finetune_config["model"]["shared_dict"]["my_type_map"] = ft_type_map

        # Add model_3 and model_4 (copies of model_2)
        finetune_config["model"]["model_dict"]["model_3"] = deepcopy(
            finetune_config["model"]["model_dict"]["model_2"]
        )
        finetune_config["model"]["model_dict"]["model_4"] = deepcopy(
            finetune_config["model"]["model_dict"]["model_2"]
        )
        finetune_config["loss_dict"]["model_3"] = deepcopy(
            finetune_config["loss_dict"]["model_2"]
        )
        finetune_config["loss_dict"]["model_4"] = deepcopy(
            finetune_config["loss_dict"]["model_2"]
        )
        finetune_config["training"]["model_prob"]["model_3"] = deepcopy(
            finetune_config["training"]["model_prob"]["model_2"]
        )
        finetune_config["training"]["model_prob"]["model_4"] = deepcopy(
            finetune_config["training"]["model_prob"]["model_2"]
        )
        finetune_config["training"]["data_dict"]["model_3"] = deepcopy(
            finetune_config["training"]["data_dict"]["model_2"]
        )
        finetune_config["training"]["data_dict"]["model_3"]["stat_file"] = (
            finetune_config["training"]["data_dict"]["model_3"]["stat_file"].replace(
                "model_2", "model_3"
            )
        )
        finetune_config["training"]["data_dict"]["model_4"] = deepcopy(
            finetune_config["training"]["data_dict"]["model_2"]
        )
        finetune_config["training"]["data_dict"]["model_4"]["stat_file"] = (
            finetune_config["training"]["data_dict"]["model_4"]["stat_file"].replace(
                "model_2", "model_4"
            )
        )

        # Set finetune rules:
        # model_1: no finetune_head → resume from model_1 (resuming=True)
        # model_2: finetune_head="model_2" → finetune from model_2
        finetune_config["model"]["model_dict"]["model_2"]["finetune_head"] = "model_2"
        # model_3: finetune_head="model_2" → finetune from model_2 (new head)
        finetune_config["model"]["model_dict"]["model_3"]["finetune_head"] = "model_2"
        # model_4: no finetune_head, new name → random fitting

        finetune_config["training"]["numb_steps"] = 1
        finetune_config["training"]["save_freq"] = 1

        finetune_config["model"], shared_links_ft = preprocess_shared_params(
            finetune_config["model"]
        )
        finetune_config["model"], finetune_links = get_finetune_rules(
            ckpt_path, finetune_config["model"]
        )
        finetune_config = update_deepmd_input(finetune_config, warning=False)
        finetune_config = normalize(finetune_config, multi_task=True)

        trainer_ft = get_trainer(
            deepcopy(finetune_config),
            finetune_model=ckpt_path,
            shared_links=shared_links_ft,
            finetune_links=finetune_links,
        )

        # Phase 4: verify weight inheritance against reference
        ft_state_dict = trainer_ft.wrapper.model.state_dict()

        # When type_map is extended, type_embedding weights for the new type
        # are randomly initialized (np.random.default_rng) during
        # change_type_map; since reference and trainer build separate
        # pretrained wrappers, these random values differ — skip them.
        _skip_type_embed = self.mixed_types

        for state_key in ft_state_dict:
            if _skip_type_embed and "type_embedding" in state_key:
                continue
            if "model_1" in state_key:
                # model_1: resume — ALL weights match reference model_1
                torch.testing.assert_close(
                    ref_state_dict[state_key],
                    ft_state_dict[state_key],
                    msg=f"model_1 (resume) weight mismatch: {state_key}",
                )
            elif (
                "model_2" in state_key
                and "out_bias" not in state_key
                and "out_std" not in state_key
            ):
                # model_2: finetune — all except out_bias/out_std
                torch.testing.assert_close(
                    ref_state_dict[state_key],
                    ft_state_dict[state_key],
                    msg=f"model_2 (finetune) weight mismatch: {state_key}",
                )
            elif (
                "model_3" in state_key
                and "out_bias" not in state_key
                and "out_std" not in state_key
            ):
                # model_3: finetune from model_2 — cross-branch key remap
                ref_key = state_key.replace("model_3", "model_2")
                torch.testing.assert_close(
                    ref_state_dict[ref_key],
                    ft_state_dict[state_key],
                    msg=f"model_3 (finetune from model_2) weight mismatch: {state_key}",
                )
            elif (
                "model_4" in state_key
                and "fitting_net" not in state_key
                and "out_bias" not in state_key
                and "out_std" not in state_key
            ):
                # model_4: random fitting — descriptor from pretrained
                # (RANDOM + from_multitask uses first pretrained key = model_1;
                # since descriptors are shared, model_1 == model_2 in pretrained)
                ref_key = state_key.replace("model_4", "model_2")
                torch.testing.assert_close(
                    ref_state_dict[ref_key],
                    ft_state_dict[state_key],
                    msg=f"model_4 (random fitting) descriptor mismatch: {state_key}",
                )

        # Phase 5: verify O/H descriptor stats are inherited from pretrained
        # (not recomputed from finetune data).
        # For mixed_types: pretrained has shape [2,...] (O,H); finetuned has
        # shape [3,...] (O,H,B). The first 2 entries must match pretrained.
        # For non-mixed_types: shapes are identical, already fully checked above.
        _STAT_SUFFIXES = ("mean", "stddev", "davg", "dstd")
        if self.mixed_types:
            n_old = len(["O", "H"])
            n_new = len(ft_type_map)
            checked_count = 0
            for key in ft_state_dict:
                if not any(key.endswith(s) for s in _STAT_SUFFIXES):
                    continue
                # Use model_1 (all branches share descriptor after share_params)
                if "model_1" not in key:
                    continue
                pre_key = key  # same key in pretrained_oh_state
                if pre_key not in pretrained_oh_state:
                    continue
                pre_val = pretrained_oh_state[pre_key]
                ft_val = ft_state_dict[key]
                # Find the type axis (size grew from n_old to n_new)
                for ax in range(pre_val.ndim):
                    if pre_val.shape[ax] == n_old and ft_val.shape[ax] == n_new:
                        for ti, tname in enumerate(["O", "H"]):
                            torch.testing.assert_close(
                                ft_val.select(ax, ti),
                                pre_val.select(ax, ti),
                                msg=(
                                    f"{tname} stat not inherited from pretrained: {key}"
                                ),
                            )
                        checked_count += 1
                        break
            self.assertGreater(
                checked_count,
                0,
                "No descriptor stat keys found for O/H inheritance check",
            )

        # Phase 6: verify case_embd inheritance (share_fitting only)
        # Pretrained branches keep their case_embd (dataset correspondence).
        # New branches (model_3 finetune from model_2, model_4 random) get
        # case_embd from the weight copy: model_3 copies model_2's, model_4
        # keeps target default (zeros since set_case_embd is skipped on finetune).
        if self.share_fitting:

            def _get_case_embd(mk):
                return trainer_ft.wrapper.model[mk].atomic_model.fitting_net.case_embd

            ce1 = _get_case_embd("model_1")
            ce2 = _get_case_embd("model_2")
            ce3 = _get_case_embd("model_3")
            ce4 = _get_case_embd("model_4")
            # Pretrained had sorted keys ["model_1","model_2"] → one-hot [1,0], [0,1]
            dim = ce1.shape[0]
            expected_eye = torch.eye(dim, dtype=ce1.dtype, device=ce1.device)
            # model_1 (resume): inherits pretrained model_1's case_embd
            torch.testing.assert_close(
                ce1,
                expected_eye[0],
                msg="model_1 case_embd should match pretrained model_1",
            )
            # model_2 (finetune from model_2): inherits pretrained model_2's case_embd
            torch.testing.assert_close(
                ce2,
                expected_eye[1],
                msg="model_2 case_embd should match pretrained model_2",
            )
            # model_3 (finetune from model_2): weight copy from model_2
            torch.testing.assert_close(
                ce3,
                expected_eye[1],
                msg="model_3 case_embd should match pretrained model_2 (finetune source)",
            )
            # model_4 (random fitting): target default (zeros, set_case_embd skipped)
            torch.testing.assert_close(
                ce4,
                torch.zeros_like(ce4),
                msg="model_4 case_embd should be zeros (random fitting, no re-init on finetune)",
            )

        # Run 1 step to verify no crash
        trainer_ft.run()

    def test_multitask_finetune_from_single_task(self) -> None:
        """Finetune multi-task model from a single-task pretrained .pt checkpoint.

        Tests the single-task pretrained → multi-task finetune path
        (finetune_from_multi_task=False, training.py:714-721).

        model_1: finetune_head="Default" → copies from single-task pretrained
        model_2: no finetune_head, not in pretrained_keys=["Default"] → RANDOM fitting
        """
        if self.share_fitting:
            # Single-task pretrained has no dim_case_embd; incompatible with
            # shared fitting multi-task target.
            return

        from deepmd.pt_expt.utils.finetune import (
            get_finetune_rules,
        )

        # Phase 1: train single-task model (2 steps)
        single_config = {
            "model": {
                "type_map": ["O", "H"],
                "descriptor": deepcopy(self.descriptor),
                "fitting_net": deepcopy(_fitting_net),
                "data_stat_nbatch": 1,
            },
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
                "training_data": {"systems": [_PT_DATA], "batch_size": 1},
                "validation_data": {
                    "systems": [_PT_DATA],
                    "batch_size": 1,
                    "numb_btch": 1,
                },
                "numb_steps": 2,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 2,
            },
        }
        single_config = update_deepmd_input(single_config, warning=False)
        single_config = normalize(single_config, multi_task=False)
        trainer_st = get_trainer(single_config)
        trainer_st.run()

        ckpt_path = os.path.join(os.getcwd(), "model.ckpt.pt")
        self.assertTrue(os.path.exists(ckpt_path), "Single-task checkpoint not created")

        # Phase 2: build reference state_dict from single-task checkpoint
        state_dict_full = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        state_dict_ckpt = (
            state_dict_full["model"] if "model" in state_dict_full else state_dict_full
        )
        pretrained_model_params = state_dict_ckpt["_extra_state"]["model_params"]

        # Single-task pretrained → wrap as {"Default": model}
        ref_model = get_model(deepcopy(pretrained_model_params)).to(DEVICE)
        pretrained_wrapper = ModelWrapper(ref_model)
        pretrained_wrapper.load_state_dict(state_dict_ckpt)
        ref_state_dict = pretrained_wrapper.model.state_dict()

        # Phase 3: build 2-branch multi-task finetune config
        finetune_config = _make_multitask_config(
            self.descriptor, share_fitting=self.share_fitting
        )
        # model_1: finetune_head="Default" → copy from single-task
        finetune_config["model"]["model_dict"]["model_1"]["finetune_head"] = "Default"
        # model_2: no finetune_head, "model_2" not in pretrained_keys=["Default"] → RANDOM
        finetune_config["training"]["numb_steps"] = 1
        finetune_config["training"]["save_freq"] = 1

        finetune_config["model"], shared_links_ft = preprocess_shared_params(
            finetune_config["model"]
        )
        finetune_config["model"], finetune_links = get_finetune_rules(
            ckpt_path, finetune_config["model"]
        )
        finetune_config = update_deepmd_input(finetune_config, warning=False)
        finetune_config = normalize(finetune_config, multi_task=True)

        trainer_ft = get_trainer(
            deepcopy(finetune_config),
            finetune_model=ckpt_path,
            shared_links=shared_links_ft,
            finetune_links=finetune_links,
        )

        # Phase 4: verify weight inheritance
        ft_state_dict = trainer_ft.wrapper.model.state_dict()

        for state_key in ft_state_dict:
            if "model_1" in state_key:
                # model_1: finetune from "Default" — all except out_bias/out_std
                if "out_bias" in state_key or "out_std" in state_key:
                    continue
                ref_key = state_key.replace("model_1", "Default")
                self.assertIn(ref_key, ref_state_dict, f"Missing ref key: {ref_key}")
                torch.testing.assert_close(
                    ref_state_dict[ref_key],
                    ft_state_dict[state_key],
                    msg=f"model_1 (from Default) weight mismatch: {state_key}",
                )
            elif "model_2" in state_key:
                if "out_bias" in state_key or "out_std" in state_key:
                    continue
                ref_key = state_key.replace("model_2", "Default")
                if ".descriptor." in state_key:
                    # Descriptor from pretrained (RANDOM uses first pretrained key)
                    self.assertIn(
                        ref_key, ref_state_dict, f"Missing ref key: {ref_key}"
                    )
                    torch.testing.assert_close(
                        ref_state_dict[ref_key],
                        ft_state_dict[state_key],
                        msg=f"model_2 (RANDOM) descriptor mismatch: {state_key}",
                    )

        # model_2 fitting NN weights (networks.*) should differ (random init)
        fitting_nn_mismatch = 0
        for state_key in ft_state_dict:
            if (
                "model_2" in state_key
                and ".fitting_net." in state_key
                and "networks" in state_key
            ):
                ref_key = state_key.replace("model_2", "Default")
                if ref_key in ref_state_dict and not torch.equal(
                    ref_state_dict[ref_key], ft_state_dict[state_key]
                ):
                    fitting_nn_mismatch += 1
        self.assertGreater(
            fitting_nn_mismatch,
            0,
            "model_2 fitting NN weights should differ from pretrained (random init)",
        )

        # Phase 5: run 1 step to verify no crash
        trainer_ft.run()

    def test_multitask_finetune_no_change_model_params(self) -> None:
        """Test change_model_params=False preserves user config in multi-task finetune.

        Contrasts with change_model_params=True which overwrites descriptor/fitting
        from pretrained (preserving trainable flags).
        """
        from deepmd.pt_expt.utils.finetune import (
            get_finetune_rules,
        )

        # Phase 1: train 2-branch multi-task model (2 steps)
        config_pretrain = _make_multitask_config(
            self.descriptor, share_fitting=self.share_fitting, numb_steps=2
        )
        config_pretrain["training"]["save_freq"] = 2
        config_pretrain["model"], shared_links_pre = preprocess_shared_params(
            config_pretrain["model"]
        )
        config_pretrain = update_deepmd_input(config_pretrain, warning=False)
        config_pretrain = normalize(config_pretrain, multi_task=True)
        trainer = get_trainer(config_pretrain, shared_links=shared_links_pre)
        trainer.run()

        ckpt_path = os.path.join(os.getcwd(), "model.ckpt.pt")
        self.assertTrue(os.path.exists(ckpt_path), "Pretrained checkpoint not created")

        # Phase 2: build finetune config with marker in descriptor
        ft_config = _make_multitask_config(
            self.descriptor, share_fitting=self.share_fitting
        )
        # model_1: no finetune_head → resume (model_1 in pretrained_keys)
        # model_2: finetune_head="model_2" → finetune
        ft_config["model"]["model_dict"]["model_2"]["finetune_head"] = "model_2"
        ft_config["training"]["numb_steps"] = 1
        ft_config["training"]["save_freq"] = 1

        # Add markers to descriptor in each branch (before preprocess_shared_params
        # resolves shared_dict references)
        ft_config["model"]["shared_dict"]["my_descriptor"]["_test_marker"] = True

        # Phase 3: test change_model_params=False
        ft_config_false = deepcopy(ft_config)
        ft_config_false["model"], _ = preprocess_shared_params(ft_config_false["model"])
        model_config_false, finetune_links_false = get_finetune_rules(
            ckpt_path, deepcopy(ft_config_false["model"]), change_model_params=False
        )

        # User config preserved: marker still present
        self.assertTrue(
            model_config_false["model_dict"]["model_1"]["descriptor"].get(
                "_test_marker", False
            ),
            "model_1 descriptor should preserve _test_marker with change_model_params=False",
        )
        self.assertTrue(
            model_config_false["model_dict"]["model_2"]["descriptor"].get(
                "_test_marker", False
            ),
            "model_2 descriptor should preserve _test_marker with change_model_params=False",
        )
        # FinetuneRuleItem has correct type_map
        for mk in ("model_1", "model_2"):
            self.assertEqual(
                finetune_links_false[mk].get_finetune_tmap(),
                ["O", "H"],
                f"{mk} finetune tmap should be ['O','H']",
            )
        # model_1 is resuming, model_2 is not
        self.assertTrue(
            finetune_links_false["model_1"].resuming,
            "model_1 should be resuming (no finetune_head, name in pretrained_keys)",
        )
        self.assertFalse(
            finetune_links_false["model_2"].resuming,
            "model_2 should not be resuming (has finetune_head)",
        )

        # Phase 4: test change_model_params=True (contrast)
        ft_config_true = deepcopy(ft_config)
        # Also set trainable=False to verify it's preserved
        ft_config_true["model"]["shared_dict"]["my_descriptor"]["trainable"] = False
        ft_config_true["model"], _ = preprocess_shared_params(ft_config_true["model"])
        model_config_true, finetune_links_true = get_finetune_rules(
            ckpt_path, deepcopy(ft_config_true["model"]), change_model_params=True
        )

        # Marker overwritten from pretrained
        self.assertFalse(
            model_config_true["model_dict"]["model_1"]["descriptor"].get(
                "_test_marker", False
            ),
            "model_1 descriptor should NOT have _test_marker with change_model_params=True",
        )
        self.assertFalse(
            model_config_true["model_dict"]["model_2"]["descriptor"].get(
                "_test_marker", False
            ),
            "model_2 descriptor should NOT have _test_marker with change_model_params=True",
        )
        # trainable=False should be preserved
        self.assertFalse(
            model_config_true["model_dict"]["model_1"]["descriptor"].get(
                "trainable", True
            ),
            "model_1 descriptor trainable should be preserved as False",
        )
        self.assertFalse(
            model_config_true["model_dict"]["model_2"]["descriptor"].get(
                "trainable", True
            ),
            "model_2 descriptor trainable should be preserved as False",
        )

        # Phase 5: build trainer with change_model_params=False → run 1 step
        ft_config_run = deepcopy(ft_config)
        ft_config_run["model"], shared_links_ft = preprocess_shared_params(
            ft_config_run["model"]
        )
        ft_config_run["model"], finetune_links_run = get_finetune_rules(
            ckpt_path, ft_config_run["model"], change_model_params=False
        )
        ft_config_run = update_deepmd_input(ft_config_run, warning=False)
        ft_config_run = normalize(ft_config_run, multi_task=True)
        trainer_ft = get_trainer(
            deepcopy(ft_config_run),
            finetune_model=ckpt_path,
            shared_links=shared_links_ft,
            finetune_links=finetune_links_run,
        )
        trainer_ft.run()

    def test_change_type_map_stat(self) -> None:
        """Validate change_type_map preserves existing types' stats.

        Tests two modes:
        1. WITHOUT model_with_new_type_stat: existing types preserved,
           new type gets default values (zeros for davg/bias, ones for dstd/std).
        2. WITH model_with_new_type_stat: existing types preserved,
           new type gets data-computed values (davg=0, dstd=0.1 for zero
           observations via StatItem defaults).
        """
        if not self.mixed_types:
            return

        old_tmap = ["O", "H"]
        new_tmap = ["O", "H", "B"]

        model_config = deepcopy(self.config["model"]["model_dict"]["model_1"])

        # Build model with old type_map and compute stats
        model = get_model(deepcopy(model_config)).to(DEVICE)
        data_systems = process_systems([_PT_DATA])
        data = DeepmdDataSystem(
            systems=data_systems,
            batch_size=1,
            test_size=1,
            type_map=old_tmap,
            trn_all_set=True,
        )
        data.add_data_requirements(_energy_data_requirement)
        model.compute_or_load_stat(
            sampled_func=lambda: make_stat_input(data, 1),
            stat_file_path=None,
        )
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}

        # ---- Test 1: change_type_map WITHOUT model_with_new_type_stat ----
        model.change_type_map(new_tmap, model_with_new_type_stat=None)
        sd_no_stat = model.state_dict()

        # Stat-like keys: descriptor mean/stddev/davg/dstd and atomic out_bias/out_std
        _STAT_SUFFIXES = ("mean", "stddev", "davg", "dstd", "out_bias", "out_std")

        def _is_stat_key(k: str) -> bool:
            return any(k.endswith(s) for s in _STAT_SUFFIXES)

        def _is_std_like(k: str) -> bool:
            return k.endswith(("stddev", "dstd", "out_std"))

        for key in sd_no_stat:
            if key not in sd_before or not _is_stat_key(key):
                continue
            old_val = sd_before[key]
            new_val = sd_no_stat[key]
            if old_val.shape == new_val.shape:
                continue
            # Find the type axis: size went from len(old_tmap) to len(new_tmap)
            for ax in range(old_val.ndim):
                if old_val.shape[ax] == len(old_tmap) and new_val.shape[ax] == len(
                    new_tmap
                ):
                    # Existing types preserved
                    torch.testing.assert_close(
                        new_val.select(ax, 0),
                        old_val.select(ax, 0),
                        msg=f"O stat changed (no model_with_new_type_stat): {key}",
                    )
                    torch.testing.assert_close(
                        new_val.select(ax, 1),
                        old_val.select(ax, 1),
                        msg=f"H stat changed (no model_with_new_type_stat): {key}",
                    )
                    # New type B: defaults (zeros for mean/davg/bias, ones for std)
                    new_B = new_val.select(ax, 2)
                    if _is_std_like(key):
                        torch.testing.assert_close(
                            new_B,
                            torch.ones_like(new_B),
                            msg=f"B default should be ones: {key}",
                        )
                    else:
                        torch.testing.assert_close(
                            new_B,
                            torch.zeros_like(new_B),
                            msg=f"B default should be zeros: {key}",
                        )
                    break

        # ---- Test 2: change_type_map WITH model_with_new_type_stat ----
        # Build fresh model with old type_map
        model2 = get_model(deepcopy(model_config)).to(DEVICE)
        model2.compute_or_load_stat(
            sampled_func=lambda: make_stat_input(data, 1),
            stat_file_path=None,
        )
        sd_before2 = {k: v.clone() for k, v in model2.state_dict().items()}

        # Build model_with_new_type_stat with extended type_map
        model_ext_config = deepcopy(model_config)
        model_ext_config["type_map"] = new_tmap
        model_ext = get_model(model_ext_config).to(DEVICE)
        data_ext = DeepmdDataSystem(
            systems=data_systems,
            batch_size=1,
            test_size=1,
            type_map=new_tmap,
            trn_all_set=True,
        )
        data_ext.add_data_requirements(_energy_data_requirement)
        model_ext.compute_or_load_stat(
            sampled_func=lambda: make_stat_input(data_ext, 1),
            stat_file_path=None,
        )

        model2.change_type_map(new_tmap, model_with_new_type_stat=model_ext)
        sd_with_stat = model2.state_dict()

        for key in sd_with_stat:
            if key not in sd_before2 or not _is_stat_key(key):
                continue
            old_val = sd_before2[key]
            new_val = sd_with_stat[key]
            if old_val.shape == new_val.shape:
                continue
            for ax in range(old_val.ndim):
                if old_val.shape[ax] == len(old_tmap) and new_val.shape[ax] == len(
                    new_tmap
                ):
                    # Existing types preserved
                    torch.testing.assert_close(
                        new_val.select(ax, 0),
                        old_val.select(ax, 0),
                        msg=f"O stat changed (with model_with_new_type_stat): {key}",
                    )
                    torch.testing.assert_close(
                        new_val.select(ax, 1),
                        old_val.select(ax, 1),
                        msg=f"H stat changed (with model_with_new_type_stat): {key}",
                    )
                    # New type B: descriptor stats should use model_ext's
                    # computed values, NOT the no-stat defaults (ones)
                    new_B = new_val.select(ax, 2)
                    is_descrpt_std = key.endswith(("stddev", "dstd"))
                    if is_descrpt_std:
                        # B has zero observations → StatItem default = 0.1
                        # (not ones like the no-stat default)
                        self.assertFalse(
                            torch.allclose(new_B, torch.ones_like(new_B)),
                            f"B descriptor stat should NOT be ones "
                            f"(should be 0.1 from StatItem default): {key}",
                        )
                    break

    def test_multitask_restart(self) -> None:
        """Train, then restart from checkpoint and verify."""
        # Phase 1: train
        config1 = deepcopy(self.config)
        config1["training"]["numb_steps"] = 2
        config1["training"]["save_freq"] = 2
        trainer1 = get_trainer(config1, shared_links=self.shared_links)
        trainer1.run()

        ckpt_path = "model.ckpt.pt"
        self.assertTrue(os.path.exists(ckpt_path), "Checkpoint not created")

        # Phase 2: restart to step 4
        config2 = deepcopy(self.config)
        config2["training"]["numb_steps"] = 4
        config2["training"]["save_freq"] = 4
        trainer2 = get_trainer(
            config2,
            restart_model=ckpt_path,
            shared_links=self.shared_links,
        )
        self.assertEqual(trainer2.start_step, 2)
        trainer2.run()

    def test_multitask_freeze(self) -> None:
        """Train, then freeze with --head and verify.

        Only runs for se_e2_a descriptor to avoid redundant slow freeze tests.
        """
        if self.descriptor.get("type") != "dpa3":
            return

        from deepmd.pt_expt.entrypoints.main import (
            freeze,
        )

        # Train
        config = deepcopy(self.config)
        trainer = get_trainer(config, shared_links=self.shared_links)
        trainer.run()

        # Freeze head model_1
        ckpt_path = "model.ckpt.pt"
        output_path = "frozen_model_1.pte"
        freeze(model=ckpt_path, output=output_path, head="model_1")
        self.assertTrue(os.path.exists(output_path), "Frozen model not created")

        # Verify frozen model loads
        from deepmd.pt_expt.model import (
            BaseModel,
        )
        from deepmd.pt_expt.utils.serialization import (
            serialize_from_file,
        )

        data = serialize_from_file(output_path)
        self.assertIn("model", data)
        frozen_model = BaseModel.deserialize(data["model"])
        self.assertIsInstance(frozen_model, torch.nn.Module)

    def test_multitask_freeze_no_head_raises(self) -> None:
        """Freezing multi-task model without --head raises ValueError.

        Only runs for se_e2_a descriptor to avoid redundant slow freeze tests.
        """
        if self.descriptor.get("type") != "dpa3":
            return

        from deepmd.pt_expt.entrypoints.main import (
            freeze,
        )

        config = deepcopy(self.config)
        trainer = get_trainer(config, shared_links=self.shared_links)
        trainer.run()

        ckpt_path = "model.ckpt.pt"
        with self.assertRaises(ValueError, msg="Should require --head"):
            freeze(model=ckpt_path, output="frozen.pte", head=None)

    def test_multitask_freeze_invalid_head_raises(self) -> None:
        """Freezing multi-task model with invalid --head raises ValueError.

        Only runs for se_e2_a descriptor to avoid redundant slow freeze tests.
        """
        if self.descriptor.get("type") != "dpa3":
            return

        from deepmd.pt_expt.entrypoints.main import (
            freeze,
        )

        config = deepcopy(self.config)
        trainer = get_trainer(config, shared_links=self.shared_links)
        trainer.run()

        ckpt_path = "model.ckpt.pt"
        with self.assertRaises(ValueError, msg="Should reject invalid head"):
            freeze(model=ckpt_path, output="frozen.pte", head="nonexistent")

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f == "lcurve.out":
                os.remove(f)
            if f.endswith(".pte"):
                os.remove(f)
        if os.path.isdir("stat_files"):
            shutil.rmtree("stat_files")


class TestMultiTaskSeA(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with se_e2_a descriptor."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_sea_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_se_e2_a
        config = _make_multitask_config(self.descriptor, share_fitting=False)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = False
        self.mixed_types = False

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskSeAShareFit(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with se_e2_a descriptor and shared fitting_net."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_sea_sf_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_se_e2_a
        config = _make_multitask_config(self.descriptor, share_fitting=True)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = True
        self.mixed_types = False

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskDPA1(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with DPA1 (se_atten) descriptor."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_dpa1_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_dpa1
        config = _make_multitask_config(self.descriptor, share_fitting=False)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = False
        self.mixed_types = True

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskDPA1ShareFit(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with DPA1 descriptor and shared fitting_net."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_dpa1_sf_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_dpa1
        config = _make_multitask_config(self.descriptor, share_fitting=True)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = True
        self.mixed_types = True

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskDPA2(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with DPA2 descriptor."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_dpa2_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_dpa2
        config = _make_multitask_config(self.descriptor, share_fitting=False)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = False
        self.mixed_types = True

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskDPA2ShareFit(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with DPA2 descriptor and shared fitting_net."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_dpa2_sf_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_dpa2
        config = _make_multitask_config(self.descriptor, share_fitting=True)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = True
        self.mixed_types = True

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskDPA3(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with DPA3 descriptor."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_dpa3_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_dpa3
        config = _make_multitask_config(self.descriptor, share_fitting=False)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = False
        self.mixed_types = True

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskDPA3ShareFit(unittest.TestCase, MultiTaskTrainTest):
    """Multi-task training with DPA3 descriptor and shared fitting_net."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_dpa3_sf_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.descriptor = _descriptor_dpa3
        config = _make_multitask_config(self.descriptor, share_fitting=True)
        config["model"], self.shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        self.config = config
        self.share_fitting = True
        self.mixed_types = True

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestMultiTaskCompile(unittest.TestCase):
    """Verify that multi-task + torch.compile works correctly."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()

    def _check_compile_correctness(self, share_fitting: bool = False) -> None:
        """Compiled multi-task model predictions and gradients match uncompiled.

        For each branch: feed the same batch through wrapper (which computes
        loss), call loss.backward(), then compare:
        1. model predictions (energy, force)
        2. loss values
        3. parameter gradients (second-order, through force loss)
        """
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        # Build uncompiled trainer
        config_uc = _make_multitask_config(
            _descriptor_se_e2_a, share_fitting=share_fitting
        )
        config_uc["model"], shared_links_uc = preprocess_shared_params(
            config_uc["model"]
        )
        config_uc = update_deepmd_input(config_uc, warning=False)
        config_uc = normalize(config_uc, multi_task=True)

        # Build compiled trainer
        config_c = _make_multitask_config(
            _descriptor_se_e2_a, share_fitting=share_fitting
        )
        config_c["training"]["enable_compile"] = True
        config_c["model"], shared_links_c = preprocess_shared_params(config_c["model"])
        config_c = update_deepmd_input(config_c, warning=False)
        config_c = normalize(config_c, multi_task=True)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_compile_corr_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            trainer_uc = get_trainer(config_uc, shared_links=shared_links_uc)
            trainer_c = get_trainer(config_c, shared_links=shared_links_c)
            for mk in ("model_1", "model_2"):
                self.assertIsInstance(trainer_c.wrapper.model[mk], _CompiledModel)

            # Copy uncompiled weights → compiled (same starting point)
            for mk in ("model_1", "model_2"):
                trainer_c.wrapper.model[mk].original_model.load_state_dict(
                    trainer_uc.wrapper.model[mk].state_dict()
                )

            # For each branch, run one forward+backward and compare
            for task_key in ("model_1", "model_2"):
                trainer_uc.optimizer.zero_grad(set_to_none=True)
                trainer_c.optimizer.zero_grad(set_to_none=True)

                input_dict, label_dict = trainer_uc.get_data(
                    is_train=True, task_key=task_key
                )

                cur_lr = trainer_uc.scheduler.get_last_lr()[0]
                pred_uc, loss_uc, _ = trainer_uc.wrapper(
                    **input_dict,
                    cur_lr=cur_lr,
                    label=label_dict,
                    task_key=task_key,
                )
                pred_c, loss_c, _ = trainer_c.wrapper(
                    **input_dict,
                    cur_lr=cur_lr,
                    label=label_dict,
                    task_key=task_key,
                )

                # Compare predictions
                torch.testing.assert_close(
                    pred_c["energy"],
                    pred_uc["energy"],
                    atol=1e-10,
                    rtol=1e-10,
                )
                torch.testing.assert_close(
                    pred_c["force"],
                    pred_uc["force"],
                    atol=1e-10,
                    rtol=1e-10,
                )
                torch.testing.assert_close(loss_c, loss_uc, atol=1e-10, rtol=1e-10)

                # Compare gradients (second-order, through force loss)
                loss_uc.backward()
                loss_c.backward()
                for (name_uc, p_uc), (name_c, p_c) in zip(
                    trainer_uc.wrapper.model[task_key].named_parameters(),
                    trainer_c.wrapper.model[task_key].original_model.named_parameters(),
                    strict=True,
                ):
                    if p_uc.grad is not None:
                        self.assertIsNotNone(
                            p_c.grad,
                            msg=f"grad is None for {name_c} (task={task_key})",
                        )
                        torch.testing.assert_close(
                            p_c.grad,
                            p_uc.grad,
                            atol=1e-10,
                            rtol=1e-10,
                            msg=f"grad mismatch on {name_uc} (task={task_key})",
                        )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compile_multitask_correctness(self) -> None:
        """Compiled multi-task predictions and gradients match uncompiled."""
        self._check_compile_correctness(share_fitting=False)

    def test_compile_multitask_correctness_share_fitting(self) -> None:
        """Compiled multi-task with shared fitting: predictions and gradients match."""
        self._check_compile_correctness(share_fitting=True)

    def test_compile_multitask_train(self) -> None:
        """Train multi-task model with torch.compile for a few steps."""
        config = _make_multitask_config(_descriptor_se_e2_a)
        config["training"]["enable_compile"] = True
        config["training"]["numb_steps"] = 2
        config["training"]["save_freq"] = 2
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_compile_train_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            trainer = get_trainer(config, shared_links=shared_links)
            trainer.run()
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compile_multitask_train_share_fitting(self) -> None:
        """Train multi-task model with shared fitting + compile for a few steps."""
        config = _make_multitask_config(_descriptor_se_e2_a, share_fitting=True)
        config["training"]["enable_compile"] = True
        config["training"]["numb_steps"] = 2
        config["training"]["save_freq"] = 2
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_compile_sf_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            trainer = get_trainer(config, shared_links=shared_links)
            trainer.run()
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Gradient accumulation test helpers
# ---------------------------------------------------------------------------


def _generate_random_data_dir(
    path: str,
    atom_types: list[int],
    nframes: int,
    seed: int,
    nfparam: int = 0,
    naparam: int = 0,
) -> None:
    """Create a minimal deepmd data directory with random data."""
    rng = np.random.RandomState(seed)
    natoms = len(atom_types)
    os.makedirs(os.path.join(path, "set.000"), exist_ok=True)

    # type.raw
    with open(os.path.join(path, "type.raw"), "w") as f:
        for t in atom_types:
            f.write(f"{t}\n")

    # box: diagonal 20x20x20
    box = np.tile(np.diag([20.0, 20.0, 20.0]).flatten(), (nframes, 1))
    np.save(os.path.join(path, "set.000", "box.npy"), box)

    # coord
    coord = rng.random((nframes, natoms * 3)) * 20.0
    np.save(os.path.join(path, "set.000", "coord.npy"), coord)

    # energy
    energy = rng.random((nframes,))
    np.save(os.path.join(path, "set.000", "energy.npy"), energy)

    # force
    force = rng.random((nframes, natoms * 3))
    np.save(os.path.join(path, "set.000", "force.npy"), force)

    # fparam (frame parameters)
    if nfparam > 0:
        fparam = rng.random((nframes, nfparam))
        np.save(os.path.join(path, "set.000", "fparam.npy"), fparam)

    # aparam (atomic parameters)
    if naparam > 0:
        aparam = rng.random((nframes, natoms * naparam))
        np.save(os.path.join(path, "set.000", "aparam.npy"), aparam)


def _make_gradient_test_mt_config(
    data_dir_0: str,
    data_dir_1: str,
    numb_fparam: int = 0,
    numb_aparam: int = 0,
) -> dict:
    """Multi-task config for gradient accumulation test."""
    type_map = ["O", "H", "C"]
    descriptor = deepcopy(_descriptor_dpa3)
    fitting_1: dict = {
        "neuron": [16, 16],
        "resnet_dt": True,
        "seed": 1,
    }
    fitting_2: dict = {
        "neuron": [16, 16],
        "resnet_dt": True,
        "seed": 2,
    }
    if numb_fparam > 0:
        fitting_1["numb_fparam"] = numb_fparam
        fitting_2["numb_fparam"] = numb_fparam
    if numb_aparam > 0:
        fitting_1["numb_aparam"] = numb_aparam
        fitting_2["numb_aparam"] = numb_aparam
    return {
        "model": {
            "shared_dict": {
                "my_type_map": type_map,
                "my_descriptor": descriptor,
            },
            "model_dict": {
                "model_1": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": fitting_1,
                    "data_stat_nbatch": 1,
                },
                "model_2": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": fitting_2,
                    "data_stat_nbatch": 1,
                },
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 0.001,
        },
        "loss_dict": {
            "model_1": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
            "model_2": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
        },
        "training": {
            "model_prob": {
                "model_1": 0.5,
                "model_2": 0.5,
            },
            "data_dict": {
                "model_1": {
                    "stat_file": "./stat_files/model_1",
                    "training_data": {
                        "systems": [data_dir_0],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir_0],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
                "model_2": {
                    "stat_file": "./stat_files/model_2",
                    "training_data": {
                        "systems": [data_dir_1],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir_1],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
            },
            "numb_steps": 2,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": 100,
        },
    }


def _make_gradient_test_st_config(
    data_dir: str,
    fitting_seed: int,
    numb_fparam: int = 0,
    numb_aparam: int = 0,
) -> dict:
    """Single-task config for gradient accumulation test."""
    type_map = ["O", "H", "C"]
    descriptor = deepcopy(_descriptor_dpa3)
    fitting: dict = {
        "neuron": [16, 16],
        "resnet_dt": True,
        "seed": fitting_seed,
    }
    if numb_fparam > 0:
        fitting["numb_fparam"] = numb_fparam
    if numb_aparam > 0:
        fitting["numb_aparam"] = numb_aparam
    return {
        "model": {
            "type_map": type_map,
            "descriptor": descriptor,
            "fitting_net": fitting,
            "data_stat_nbatch": 1,
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 0.001,
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
                "systems": [data_dir],
                "batch_size": 1,
            },
            "validation_data": {
                "systems": [data_dir],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": 1,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": 100,
        },
    }


def _deterministic_task_choice(task_sequence: list[int]):
    """Return a patched dp_random.choice that forces task selection order."""
    original = dp_random.choice
    it = iter(task_sequence)

    def patched(a, size=None, replace=True, p=None):
        # Task selection: array with >=2 elements and probability vector
        if hasattr(a, "__len__") and len(a) >= 2 and p is not None:
            return next(it)
        return original(a, size=size, replace=replace, p=p)

    return patched


def _make_recording_step(
    trainer,
    modules_to_record: dict,
    recorded_grads: list[dict],
):
    """Patch _optimizer_step: record grads from listed modules, skip optimizer.

    Parameters
    ----------
    trainer : Trainer
        The trainer whose scheduler.step() is called.
    modules_to_record : dict[str, torch.nn.Module]
        Named modules whose parameter gradients to record.
    recorded_grads : list[dict[str, torch.Tensor]]
        Appended with {module_key/param_name: grad} at each step.
    """

    def recording_step():
        grads = {}
        for mod_key, mod in modules_to_record.items():
            for n, p in mod.named_parameters():
                if p.grad is not None:
                    grads[f"{mod_key}/{n}"] = p.grad.clone()
        recorded_grads.append(grads)
        trainer.scheduler.step()

    return recording_step


class TestMultiTaskGradient(unittest.TestCase):
    """Verify multi-task descriptor gradients match sum of single-task gradients."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_grad_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        self.nfparam = 2
        self.naparam = 3

        self.data_dir_0 = os.path.join(self.tmpdir, "data_task0")
        _generate_random_data_dir(
            self.data_dir_0,
            atom_types=[0, 0, 1, 1, 1, 2],
            nframes=1,
            seed=42,
            nfparam=self.nfparam,
            naparam=self.naparam,
        )
        self.data_dir_1 = os.path.join(self.tmpdir, "data_task1")
        _generate_random_data_dir(
            self.data_dir_1,
            atom_types=[0, 1, 1, 2, 2, 2, 2],
            nframes=1,
            seed=137,
            nfparam=self.nfparam,
            naparam=self.naparam,
        )

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_gradient_accumulation(self) -> None:
        """Sum of per-task grads from multi-task run() ==
        sum of grads from two single-task run() calls.
        """
        # ===== Multi-task trainer =====
        mt_config = _make_gradient_test_mt_config(
            self.data_dir_0,
            self.data_dir_1,
            numb_fparam=self.nfparam,
            numb_aparam=self.naparam,
        )
        mt_config["model"], shared_links = preprocess_shared_params(mt_config["model"])
        mt_config = update_deepmd_input(mt_config, warning=False)
        mt_config = normalize(mt_config, multi_task=True)

        mt_trainer = get_trainer(deepcopy(mt_config), shared_links=shared_links)
        mt_desc = mt_trainer.wrapper.model["model_1"].atomic_model.descriptor
        mt_fit_1 = mt_trainer.wrapper.model["model_1"].atomic_model.fitting_net
        mt_fit_2 = mt_trainer.wrapper.model["model_2"].atomic_model.fitting_net

        # Verify descriptor params are aliased (share_params)
        mt_desc_2 = mt_trainer.wrapper.model["model_2"].atomic_model.descriptor
        for (n1, p1), (n2, p2) in zip(
            mt_desc.named_parameters(), mt_desc_2.named_parameters(), strict=True
        ):
            assert p1.data_ptr() == p2.data_ptr(), (
                f"Descriptor params not aliased: {n1}"
            )

        # Record grads for descriptor + both fitting heads
        mt_grads: list[dict[str, torch.Tensor]] = []
        mt_trainer._optimizer_step = _make_recording_step(
            mt_trainer,
            {"desc": mt_desc, "fit_1": mt_fit_1, "fit_2": mt_fit_2},
            mt_grads,
        )
        with patch(
            "deepmd.utils.random.choice",
            _deterministic_task_choice([0, 1]),
        ):
            mt_trainer.run()  # 2 steps: task_0 then task_1

        assert len(mt_grads) == 2

        # ===== Single-task trainer for task_0 =====
        st0_config = _make_gradient_test_st_config(
            self.data_dir_0,
            fitting_seed=1,  # same as model_1
            numb_fparam=self.nfparam,
            numb_aparam=self.naparam,
        )
        st0_config = update_deepmd_input(st0_config, warning=False)
        st0_config = normalize(st0_config)

        os.chdir(tempfile.mkdtemp(dir=self.tmpdir))  # fresh cwd
        st0_trainer = get_trainer(deepcopy(st0_config))

        # Copy MT model_1 state → ST0 to ensure identical params+buffers
        # (stat buffers like davg/dstd/bias_atom_e differ due to data)
        mt_m1 = mt_trainer.wrapper.model["model_1"]
        st0_m = st0_trainer.wrapper.model["Default"]
        st0_m.load_state_dict(mt_m1.state_dict())

        st0_desc = st0_m.atomic_model.descriptor
        st0_fit = st0_m.atomic_model.fitting_net

        st0_grads: list[dict[str, torch.Tensor]] = []
        st0_trainer._optimizer_step = _make_recording_step(
            st0_trainer, {"desc": st0_desc, "fit": st0_fit}, st0_grads
        )
        st0_trainer.run()  # 1 step
        assert len(st0_grads) == 1

        # ===== Single-task trainer for task_1 =====
        st1_config = _make_gradient_test_st_config(
            self.data_dir_1,
            fitting_seed=2,  # same as model_2
            numb_fparam=self.nfparam,
            numb_aparam=self.naparam,
        )
        st1_config = update_deepmd_input(st1_config, warning=False)
        st1_config = normalize(st1_config)

        os.chdir(tempfile.mkdtemp(dir=self.tmpdir))  # fresh cwd
        st1_trainer = get_trainer(deepcopy(st1_config))

        # Copy MT model_2 state → ST1 to ensure identical params+buffers
        mt_m2 = mt_trainer.wrapper.model["model_2"]
        st1_m = st1_trainer.wrapper.model["Default"]
        st1_m.load_state_dict(mt_m2.state_dict())

        st1_desc = st1_m.atomic_model.descriptor
        st1_fit = st1_m.atomic_model.fitting_net

        st1_grads: list[dict[str, torch.Tensor]] = []
        st1_trainer._optimizer_step = _make_recording_step(
            st1_trainer, {"desc": st1_desc, "fit": st1_fit}, st1_grads
        )
        st1_trainer.run()  # 1 step
        assert len(st1_grads) == 1

        # ===== Comparison: descriptor gradients =====
        # Multi-task descriptor grad at each step should match single-task
        desc_keys = [k for k in mt_grads[0] if k.startswith("desc/")]
        assert len(desc_keys) > 0, "No descriptor gradients"

        # Per-task descriptor grad: mt step_0 == st_0, mt step_1 == st_1
        for name in desc_keys:
            np.testing.assert_allclose(
                mt_grads[0][name].detach().cpu().numpy(),
                st0_grads[0][name].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Task_0 descriptor grad mismatch: {name}",
            )
            np.testing.assert_allclose(
                mt_grads[1][name].detach().cpu().numpy(),
                st1_grads[0][name].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Task_1 descriptor grad mismatch: {name}",
            )

        # Descriptor grad sum: mt(step0 + step1) == st0 + st1
        for name in desc_keys:
            mt_sum = mt_grads[0][name] + mt_grads[1][name]
            st_sum = st0_grads[0][name] + st1_grads[0][name]
            np.testing.assert_allclose(
                mt_sum.detach().cpu().numpy(),
                st_sum.detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Descriptor grad sum mismatch: {name}",
            )

        # ===== Comparison: fitting head gradients =====
        # Step 0 trains model_1 → mt fit_1 grads == st0 fit grads
        fit1_keys = [k for k in mt_grads[0] if k.startswith("fit_1/")]
        for name in fit1_keys:
            st_name = name.replace("fit_1/", "fit/")
            np.testing.assert_allclose(
                mt_grads[0][name].detach().cpu().numpy(),
                st0_grads[0][st_name].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Fitting head grad mismatch (task_0): {name}",
            )
        # Verify fit_2 not in step 0 grads (not part of computation graph)
        assert not any(k.startswith("fit_2/") for k in mt_grads[0]), (
            "fit_2 should have no gradients in step 0 (task_0)"
        )

        # Step 1 trains model_2 → mt fit_2 grads == st1 fit grads
        fit2_keys = [k for k in mt_grads[1] if k.startswith("fit_2/")]
        for name in fit2_keys:
            st_name = name.replace("fit_2/", "fit/")
            np.testing.assert_allclose(
                mt_grads[1][name].detach().cpu().numpy(),
                st1_grads[0][st_name].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Fitting head grad mismatch (task_1): {name}",
            )
        # Verify fit_1 not in step 1 grads
        assert not any(k.startswith("fit_1/") for k in mt_grads[1]), (
            "fit_1 should have no gradients in step 1 (task_1)"
        )


class TestCompileCaseEmbdVaryingNframes(unittest.TestCase):
    """Compiled multi-task with ``dim_case_embd > 0`` and varying ``nframes``.

    The shared-fitting path in ``GeneralFitting.call`` tiles the per-task
    case embedding as ``xp.tile(reshape(case_embd, (1, 1, -1)), (nf, nloc, 1))``
    (see ``deepmd/dpmodel/fitting/general_fitting.py``).  Under
    ``tracing_mode="symbolic"`` the ``nf`` multiplier must stay symbolic;
    otherwise the compiled graph hard-codes a specific batch size and
    subsequent calls with a different ``nframes`` error out.

    The test uses two systems with different atom counts and per-system
    ``batch_size=[2, 3]`` so every branch's compiled graph sees both
    nframes values.  ``dim_case_embd=2`` is deliberately chosen to also
    collide numerically with the nframes=2 runtime case.  ``dp_random.choice``
    is mocked so both tasks and both systems are sampled.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_case_embd_vary_")
        cls.sys0_m1 = os.path.join(cls.tmpdir, "sys0_model1_6atoms")
        cls.sys1_m1 = os.path.join(cls.tmpdir, "sys1_model1_4atoms")
        cls.sys0_m2 = os.path.join(cls.tmpdir, "sys0_model2_6atoms")
        cls.sys1_m2 = os.path.join(cls.tmpdir, "sys1_model2_4atoms")
        for path, seed in (
            (cls.sys0_m1, 11),
            (cls.sys1_m1, 12),
            (cls.sys0_m2, 21),
            (cls.sys1_m2, 22),
        ):
            _generate_random_data_dir(
                path,
                atom_types=[i % 2 for i in range(6 if "6atoms" in path else 4)],
                nframes=4,
                seed=seed,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _make_config(self) -> dict:
        type_map = ["O", "H"]
        fitting = deepcopy(_fitting_net)
        fitting["dim_case_embd"] = 2
        shared_dict: dict = {
            "my_type_map": type_map,
            "my_descriptor": deepcopy(_descriptor_se_e2_a),
            "my_fitting": fitting,
        }
        config = {
            "model": {
                "shared_dict": shared_dict,
                "model_dict": {
                    "model_1": {
                        "type_map": "my_type_map",
                        "descriptor": "my_descriptor",
                        "fitting_net": "my_fitting",
                        "data_stat_nbatch": 1,
                    },
                    "model_2": {
                        "type_map": "my_type_map",
                        "descriptor": "my_descriptor",
                        "fitting_net": "my_fitting",
                        "data_stat_nbatch": 1,
                    },
                },
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": 500,
                "start_lr": 0.001,
                "stop_lr": 3.51e-8,
            },
            "loss_dict": {
                "model_1": {
                    "type": "ener",
                    "start_pref_e": 0.02,
                    "limit_pref_e": 1,
                    "start_pref_f": 1000,
                    "limit_pref_f": 1,
                    "start_pref_v": 0,
                    "limit_pref_v": 0,
                },
                "model_2": {
                    "type": "ener",
                    "start_pref_e": 0.02,
                    "limit_pref_e": 1,
                    "start_pref_f": 1000,
                    "limit_pref_f": 1,
                    "start_pref_v": 0,
                    "limit_pref_v": 0,
                },
            },
            "training": {
                "enable_compile": True,
                "model_prob": {"model_1": 0.5, "model_2": 0.5},
                "data_dict": {
                    "model_1": {
                        "stat_file": "./stat_files/model_1",
                        "training_data": {
                            "systems": [self.sys0_m1, self.sys1_m1],
                            "batch_size": [2, 3],
                        },
                        "validation_data": {
                            "systems": [self.sys0_m1],
                            "batch_size": 1,
                            "numb_btch": 1,
                        },
                    },
                    "model_2": {
                        "stat_file": "./stat_files/model_2",
                        "training_data": {
                            "systems": [self.sys0_m2, self.sys1_m2],
                            "batch_size": [2, 3],
                        },
                        "validation_data": {
                            "systems": [self.sys0_m2],
                            "batch_size": 1,
                            "numb_btch": 1,
                        },
                    },
                },
                "numb_steps": 1,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 100,
                "save_freq": 100,
            },
        }
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        return config, shared_links

    def test_compiled_varying_nframes_with_case_embd(self) -> None:
        """Compiled shared-fitting graph handles nframes in {2, 3} per branch."""
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        config, shared_links = self._make_config()
        tmpdir = tempfile.mkdtemp(prefix="pt_expt_mt_case_embd_run_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            trainer = get_trainer(deepcopy(config), shared_links=shared_links)
            # Both branches must be compiled.
            for mk in ("model_1", "model_2"):
                self.assertIsInstance(trainer.wrapper.model[mk], _CompiledModel)
                ce = trainer.wrapper.model[
                    mk
                ].original_model.atomic_model.fitting_net.case_embd
                self.assertIsNotNone(ce, f"case_embd not set on {mk}")
                self.assertEqual(int(ce.shape[0]), 2)

            # Drive 6 steps alternating (task, system_index) so each branch's
            # compiled graph sees both nframes=2 (sys0) and nframes=3 (sys1).
            trainer.wrapper.train()
            task_sequence = ["model_1", "model_2"] * 3
            sys_sequence = [0, 1, 0, 1, 0, 1]
            sys_iter = iter(sys_sequence)

            original_choice = dp_random.choice

            def task_or_system_choice(a, size=None, replace=True, p=None):
                # Per-branch system selection: alternate between the two
                # systems so every compiled graph sees both nframes values.
                if hasattr(a, "__len__") and len(a) == 2 and p is not None:
                    return next(sys_iter)
                return original_choice(a, size=size, replace=replace, p=p)

            seen_nframes: set[int] = set()
            with patch.object(dp_random, "choice", side_effect=task_or_system_choice):
                for task_key in task_sequence:
                    trainer.optimizer.zero_grad(set_to_none=True)
                    inp, lab = trainer.get_data(is_train=True, task_key=task_key)
                    seen_nframes.add(int(inp["coord"].shape[0]))
                    lr = trainer.scheduler.get_last_lr()[0]
                    _, loss, _ = trainer.wrapper(
                        **inp, cur_lr=lr, label=lab, task_key=task_key
                    )
                    loss.backward()
                    trainer.optimizer.step()
                    self.assertFalse(torch.isnan(loss), "loss is NaN")
                    self.assertFalse(torch.isinf(loss), "loss is Inf")

            self.assertEqual(
                seen_nframes,
                {2, 3},
                msg=(
                    f"nframes did not vary across steps: {seen_nframes}. "
                    "Expected both 2 and 3 (matching and not matching dim_case_embd=2)."
                ),
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
