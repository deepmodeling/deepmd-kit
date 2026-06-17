# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify emitted input.json matches the MFT paper repo
(Chengqian-Zhang/Multitask-finetuning/examples/qm9_gap/).

Covers single-task FT/LP/Scratch (DPATrainer) and multi-task property-mode
MFT (MFTConfigManager). Configs are round-tripped through json to confirm
the fields survive serialization (no GPU / no real ckpt needed).

Backward-compat note: legacy ener-mode MFT (mp_data sensitivity analysis)
must stay byte-for-byte unchanged; that is locked by
test_ener_mode_byte_for_byte_unchanged.
"""

from __future__ import (
    annotations,
)

import json
from unittest.mock import (
    patch,
)

from dpa_adapt.config.manager import (
    MFTConfigManager,
)
from dpa_adapt.trainer import (
    DPATrainer,
)

TYPE_MAP = ["H", "C", "N", "O"]


def _make_sys(tmp_path) -> str:
    """Create one real system dir and return a glob matching it (DPATrainer
    expands globs against the filesystem).
    """
    root = tmp_path / "sys"
    root.mkdir(parents=True, exist_ok=True)
    (root / "s_000").mkdir(exist_ok=True)
    return str(root / "s_*")


# ---------------------------------------------------------------------------
# DPATrainer (FT / LP / Scratch) helpers
# ---------------------------------------------------------------------------


def _fake_descriptor_sd() -> dict:
    """Checkpoint state_dict shaped like DPA-3.1-3M: a custom_silu descriptor
    with no fix_stat_std, to prove _get_descriptor overrides both.
    """
    descriptor = {
        "type": "dpa3",
        "repflow": {"n_dim": 128, "e_dim": 64, "a_dim": 32, "nlayers": 16},
        "activation_function": "custom_silu:3.0",
        "precision": "float32",
        "trainable": True,
    }
    return {
        "model": {
            "_extra_state": {
                "model_params": {"shared_dict": {"dpa3_descriptor": descriptor}}
            }
        }
    }


def _patch_torch_load():
    return patch("torch.load", lambda *a, **kw: _fake_descriptor_sd())


def _trainer(pretrained, tmp_path, **overrides):
    sys_glob = _make_sys(tmp_path)
    kwargs = dict(
        pretrained=pretrained,
        train_systems=sys_glob,
        valid_systems=sys_glob,
        type_map=TYPE_MAP,
    )
    kwargs.update(overrides)
    return DPATrainer(**kwargs)


def _lp_config(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    t = _trainer(
        str(ckpt), tmp_path, freeze_backbone=True, output_dir=str(tmp_path / "o")
    )
    with _patch_torch_load():
        config = t._build_config()
    # Round-trip through json to mirror how fit() writes input.json.
    return json.loads(json.dumps(config)), t


def _ft_config(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    t = _trainer(
        str(ckpt), tmp_path, freeze_backbone=False, output_dir=str(tmp_path / "o")
    )
    with _patch_torch_load():
        config = t._build_config()
    return json.loads(json.dumps(config)), t


# ---------------------------------------------------------------------------
# LP single-task input.json
# ---------------------------------------------------------------------------


def test_lp_input_json_no_dim_case_embd(tmp_path):
    config, _ = _lp_config(tmp_path)
    assert "dim_case_embd" not in config["model"]["fitting_net"]


def test_lp_input_json_descriptor_trainable_false(tmp_path):
    config, _ = _lp_config(tmp_path)
    assert config["model"]["descriptor"]["trainable"] is False


def test_lp_input_json_activation_silut(tmp_path):
    config, _ = _lp_config(tmp_path)
    assert config["model"]["descriptor"]["activation_function"] == "silut:3.0"


def test_lp_input_json_fix_stat_std_0_3(tmp_path):
    config, _ = _lp_config(tmp_path)
    assert config["model"]["descriptor"]["repflow"]["fix_stat_std"] == 0.3


def test_lp_input_json_decay_steps_1000(tmp_path):
    config, _ = _lp_config(tmp_path)
    assert config["learning_rate"]["decay_steps"] == 1000


def test_lp_input_json_gradient_max_norm_5(tmp_path):
    config, _ = _lp_config(tmp_path)
    assert config["training"]["gradient_max_norm"] == 5.0


def test_lp_cmd_no_model_branch_flag(tmp_path):
    _, t = _lp_config(tmp_path)
    cmd = t._build_cmd("input.json")
    assert "--model-branch" not in cmd
    assert "--finetune" in cmd
    assert "--skip-neighbor-stat" in cmd


def test_lp_input_json_loss_is_property(tmp_path):
    config, _ = _lp_config(tmp_path)
    loss = config["loss"]
    assert loss["type"] == "property"
    assert loss["loss_func"] == "mse"
    assert loss["metric"] == ["mae", "rmse"]


# ---------------------------------------------------------------------------
# FT single-task input.json
# ---------------------------------------------------------------------------


def test_ft_input_json_descriptor_trainable_true(tmp_path):
    """FT (freeze_backbone=False) keeps the descriptor trainable; paper FT
    input.json omits trainable (defaults true). We emit trainable=true, which
    is the same effective config.
    """
    config, _ = _ft_config(tmp_path)
    assert config["model"]["descriptor"]["trainable"] is True


def test_ft_input_json_no_dim_case_embd(tmp_path):
    config, _ = _ft_config(tmp_path)
    assert "dim_case_embd" not in config["model"]["fitting_net"]


def test_ft_cmd_no_model_branch_flag(tmp_path):
    _, t = _ft_config(tmp_path)
    cmd = t._build_cmd("input.json")
    assert "--model-branch" not in cmd
    assert "--finetune" in cmd


# ---------------------------------------------------------------------------
# MFT multi-task property-mode input.json
# ---------------------------------------------------------------------------


class _PropertyTuner:
    pretrained = "/share/DPA-3.1-3M.pt"
    aux_branch = "SPICE2"
    aux_prob = 0.5
    type_map = ["H", "C", "N", "O"]
    fitting_net_params = {
        "type": "ener",
        "neuron": [240, 240, 240],
        "dim_case_embd": 31,
        "seed": 1,
    }
    downstream_task_type = "property"
    property_name = "homo"
    task_dim = 1
    intensive = True
    learning_rate = 1e-3
    stop_lr = 1e-5
    max_steps = 100000
    batch_size = "auto:32"
    seed = 42
    output_dir = "/tmp/mft_paper"
    save_freq = 500
    disp_freq = 100
    train_data = "/data/qm9"
    aux_data = "/data/spice2"
    valid_data = None


def _mft_property_config():
    return json.loads(json.dumps(MFTConfigManager(_PropertyTuner()).build()))


def test_mft_input_json_downstream_branch_key_is_property():
    """Paper repo names the downstream branch "property" (not "DOWNSTREAM")
    across model_dict / loss_dict / model_prob / data_dict.
    """
    config = _mft_property_config()
    md = config["model"]["model_dict"]
    assert "property" in md
    assert "DOWNSTREAM" not in md
    assert "property" in config["loss_dict"]
    assert "property" in config["training"]["model_prob"]
    assert "property" in config["training"]["data_dict"]


def test_mft_input_json_downstream_finetune_head_random():
    config = _mft_property_config()
    assert config["model"]["model_dict"]["property"]["finetune_head"] == "RANDOM"


def test_mft_input_json_aux_finetune_head_branch_name():
    config = _mft_property_config()
    assert config["model"]["model_dict"]["SPICE2"]["finetune_head"] == "SPICE2"


def test_mft_input_json_downstream_has_dim_case_embd():
    config = _mft_property_config()
    fn = config["model"]["model_dict"]["property"]["fitting_net"]
    assert fn["dim_case_embd"] == 31


def test_mft_input_json_aux_keeps_dim_case_embd():
    config = _mft_property_config()
    fn = config["model"]["model_dict"]["SPICE2"]["fitting_net"]
    assert fn["dim_case_embd"] == 31


def test_mft_input_json_property_mode_loss_is_property():
    config = _mft_property_config()
    loss = config["loss_dict"]["property"]
    assert loss["type"] == "property"
    assert loss["loss_func"] == "mse"
    # aux branch keeps ener loss
    assert config["loss_dict"]["SPICE2"]["type"] == "ener"


def test_mft_input_json_descriptor_silut_and_fix_stat_std():
    config = _mft_property_config()
    desc = config["model"]["shared_dict"]["dpa3_descriptor"]
    assert desc["activation_function"] == "silut:3.0"
    assert desc["repflow"]["fix_stat_std"] == 0.3


def test_mft_input_json_decay_steps_1000_and_grad_norm():
    config = _mft_property_config()
    assert config["learning_rate"]["decay_steps"] == 1000
    assert config["training"]["gradient_max_norm"] == 5.0


def test_mft_input_json_batch_sizes():
    config = _mft_property_config()
    dd = config["training"]["data_dict"]
    assert dd["SPICE2"]["training_data"]["batch_size"] == "auto:128"
    assert dd["property"]["training_data"]["batch_size"] == "auto:512"


def test_mft_input_json_model_prob_default_half_half():
    config = _mft_property_config()
    prob = config["training"]["model_prob"]
    assert prob["SPICE2"] == 0.5
    assert prob["property"] == 0.5


def test_mft_cmd_no_model_branch():
    cm = MFTConfigManager(_PropertyTuner())
    cmd = cm.build_cmd("input.json")
    assert "--model-branch" not in cmd
    assert "--finetune" in cmd


# ---------------------------------------------------------------------------
# Backward compat: legacy ener-mode MFT must be byte-for-byte unchanged
# ---------------------------------------------------------------------------


class _EnerTuner:
    """No downstream_task_type attr — legacy mp_data sensitivity-analysis
    caller. Must produce the pre-paper-alignment config exactly.
    """

    pretrained = "/share/DPA-3.1-3M.pt"
    aux_branch = "MP_traj_v024_alldata_mixu"
    aux_prob = 0.5
    type_map = ["Cu", "O"]
    fitting_net_params = {"type": "ener", "neuron": [240, 240, 240]}
    learning_rate = 1e-3
    stop_lr = 1e-5
    max_steps = 1000
    batch_size = "auto:32"
    seed = 42
    output_dir = "/tmp/mft_ener"
    save_freq = 500
    disp_freq = 100
    train_data = "/data/downstream"
    aux_data = "/data/aux"
    valid_data = None


# The expected legacy config, frozen from the pre-2026-05-20 manager.py output.
_LEGACY_ENER_EXPECTED = {
    "model": {
        "shared_dict": {
            "dpa3_descriptor": {
                "type": "dpa3",
                "repflow": {
                    "n_dim": 128,
                    "e_dim": 64,
                    "a_dim": 32,
                    "nlayers": 16,
                    "e_rcut": 6.0,
                    "e_rcut_smth": 5.3,
                    "e_sel": 1200,
                    "a_rcut": 4.0,
                    "a_rcut_smth": 3.5,
                    "a_sel": 300,
                    "axis_neuron": 4,
                    "skip_stat": True,
                    "a_compress_rate": 1,
                    "a_compress_e_rate": 2,
                    "a_compress_use_split": True,
                    "update_angle": True,
                    "smooth_edge_update": True,
                    "use_dynamic_sel": True,
                    "sel_reduce_factor": 10.0,
                    "update_style": "res_residual",
                    "update_residual": 0.1,
                    "update_residual_init": "const",
                    "n_multi_edge_message": 1,
                    "optim_update": True,
                    "use_exp_switch": True,
                },
                "activation_function": "custom_silu:3.0",
                "precision": "float32",
                "use_tebd_bias": False,
                "concat_output_tebd": False,
                "exclude_types": [],
                "env_protection": 0.0,
                "trainable": True,
                "use_econf_tebd": False,
            },
            "type_map": ["Cu", "O"],
        },
        "model_dict": {
            "MP_traj_v024_alldata_mixu": {
                "type_map": "type_map",
                "descriptor": "dpa3_descriptor",
                "fitting_net": {"type": "ener", "neuron": [240, 240, 240]},
            },
            "DOWNSTREAM": {
                "finetune_head": "MP_traj_v024_alldata_mixu",
                "type_map": "type_map",
                "descriptor": "dpa3_descriptor",
                "fitting_net": {"type": "ener", "neuron": [240, 240, 240]},
            },
        },
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 1e-3,
        "stop_lr": 1e-5,
        "decay_steps": 5000,
    },
    "loss_dict": {
        "MP_traj_v024_alldata_mixu": {
            "type": "ener",
            "start_pref_e": 0.2,
            "limit_pref_e": 20,
            "start_pref_f": 100,
            "limit_pref_f": 60,
            "start_pref_v": 0.02,
            "limit_pref_v": 1,
        },
        "DOWNSTREAM": {
            "type": "ener",
            "start_pref_e": 0.2,
            "limit_pref_e": 20,
            "start_pref_f": 100,
            "limit_pref_f": 60,
            "start_pref_v": 0.02,
            "limit_pref_v": 1,
        },
    },
    "training": {
        "model_prob": {"MP_traj_v024_alldata_mixu": 0.5, "DOWNSTREAM": 1.0},
        "data_dict": {
            "MP_traj_v024_alldata_mixu": {
                "training_data": {"systems": ["/data/aux"], "batch_size": "auto:32"}
            },
            "DOWNSTREAM": {
                "training_data": {
                    "systems": ["/data/downstream"],
                    "batch_size": "auto:32",
                }
            },
        },
        "numb_steps": 1000,
        "save_freq": 500,
        "disp_freq": 100,
        "seed": 42,
    },
}


def test_ener_mode_byte_for_byte_unchanged():
    """Legacy ener MFT config (and its JSON serialization) must equal the
    frozen pre-paper-alignment output exactly — including key order.
    """
    config = MFTConfigManager(_EnerTuner()).build()
    assert config == _LEGACY_ENER_EXPECTED
    # Byte-for-byte JSON (key order preserved by Python dict insertion order).
    assert json.dumps(config) == json.dumps(_LEGACY_ENER_EXPECTED)


def test_ener_mode_no_gradient_max_norm():
    config = MFTConfigManager(_EnerTuner()).build()
    assert "gradient_max_norm" not in config["training"]


def test_ener_mode_no_fix_stat_std():
    config = MFTConfigManager(_EnerTuner()).build()
    assert "fix_stat_std" not in config["model"]["shared_dict"]["dpa3_descriptor"]
