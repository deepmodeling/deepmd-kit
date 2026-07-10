# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
from typing import (
    Any,
)

from dpa_adapt._backend import (
    resolve_dp_command,
)
from dpa_adapt.mft import (
    _PROPERTY_LIKE_DOWNSTREAM_TYPES,
)


def _aux_dim_case_embd(t: Any) -> int:
    """dim_case_embd required on the DOWNSTREAM head to share a descriptor
    with the aux branch.

    Read from the aux branch's own (ckpt-derived) ``fitting_net_params``
    rather than hardcoded: it must equal the branch count of whatever
    multi-task checkpoint the aux branch was itself pretrained as part of
    (deepmd-kit's multi-task trainer requires every model_dict branch to
    declare the same dim_case_embd -- see
    deepmd.pt.train.training.get_case_embd_config). That count is 31 for
    DPA-3.1-3M but differs for other checkpoints (e.g. 23 for the
    OMol25/Organic_Reactions/ODAC23 checkpoint); hardcoding 31 silently
    mismatches every checkpoint that isn't DPA-3.1-3M. Falls back to 0 (no
    case embedding, matching the aux branch) when the aux fitting_net has
    none, e.g. a single-task-pretrained checkpoint.
    """
    return (getattr(t, "fitting_net_params", None) or {}).get("dim_case_embd", 0)


# Default property-head architecture for MFT DOWNSTREAM when
# downstream_task_type="property". Mirrors DPATrainer.DEFAULT_FITTING_NET
# (trainer.py L64-70). dim_case_embd is added dynamically by
# _build_property_fitting_net via _aux_dim_case_embd, not baked in here:
# DPATrainer is single-task and doesn't need this field at all, while MFT's
# correct value depends on which checkpoint is being finetuned.
_PROPERTY_FITTING_NET_BASE = {
    "type": "property",
    "neuron": [240, 240, 240],
    "activation_function": "tanh",
    "resnet_dt": True,
    "precision": "float32",
}


def _build_property_fitting_net(t: Any) -> dict:
    """Construct a property fitting_net dict from a tuner's property params.
    The property head is independent of the aux branch's ener fitting_net
    that came out of the ckpt — reusing the ener config silently introduces
    a force-field bias layer (Bug root cause).
    """
    fn = dict(_PROPERTY_FITTING_NET_BASE)
    fn.update(
        {
            "property_name": t.property_name,
            "task_dim": t.task_dim,
            "intensive": t.intensive,
            "seed": t.seed,
        }
    )
    dim_case_embd = _aux_dim_case_embd(t)
    if dim_case_embd:
        fn["dim_case_embd"] = dim_case_embd
    if getattr(t, "fparam_dim", 0) > 0:
        fn["numb_fparam"] = t.fparam_dim
    return fn


def _build_property_loss() -> dict:
    """Property-task loss for DOWNSTREAM. Notes:
    - No start_pref_f / start_pref_v: HOMO/LUMO data has no forces/virials.
    - property_name MUST NOT appear here: deepmd 3.1.3 strict-mode dargs
      rejects unknown keys inside loss_property (it belongs on fitting_net).
    """
    return {
        "type": "property",
        "loss_func": "mse",
        "metric": ["mae", "rmse"],
        "beta": 1.0,
    }


# Default group_property-head architecture for MFT DOWNSTREAM when
# downstream_task_type="group_property". Independent of
# _PROPERTY_FITTING_NET_BASE (not a trimmed copy of it): GroupPropertyFittingNet
# is a small standalone MLP, not built on GeneralFitting, so several
# property-schema fields (resnet_dt, intensive, ...) don't exist on it and
# dargs strict-mode rejects them outright rather than ignoring them -- see
# deepmd.utils.argcheck.fitting_group_property. dim_case_embd is added
# dynamically by _build_group_property_fitting_net via _aux_dim_case_embd,
# not baked in here, for the same reason as the property head.
_GROUP_PROPERTY_FITTING_NET_BASE = {
    "type": "group_property",
    "neuron": [240, 240, 240],
    "activation_function": "gelu",
    "precision": "float32",
}


def _build_group_property_fitting_net(t: Any) -> dict:
    """Construct a group_property fitting_net dict from a tuner's params."""
    fn = dict(_GROUP_PROPERTY_FITTING_NET_BASE)
    fn.update(
        {
            "property_name": t.property_name,
            "task_dim": t.task_dim,
            "group_reduce": getattr(t, "group_reduce", "mean"),
            "seed": t.seed,
        }
    )
    dim_case_embd = _aux_dim_case_embd(t)
    if dim_case_embd:
        fn["dim_case_embd"] = dim_case_embd
    if getattr(t, "fparam_dim", 0) > 0:
        fn["numb_fparam"] = t.fparam_dim
    return fn


def _build_group_property_loss() -> dict:
    """group_property-task loss for DOWNSTREAM.

    deepmd.utils.argcheck.loss_group_property() reuses loss_property()'s
    schema verbatim, so this only differs from _build_property_loss() by
    ``type``.
    """
    return {
        "type": "group_property",
        "loss_func": "mse",
        "metric": ["mae", "rmse"],
        "beta": 1.0,
    }


_ENER_LOSS = {
    "type": "ener",
    "start_pref_e": 0.2,
    "limit_pref_e": 20,
    "start_pref_f": 100,
    "limit_pref_f": 60,
    "start_pref_v": 0.02,
    "limit_pref_v": 1,
}


class MFTConfigManager:
    def __init__(self, tuner: Any) -> None:
        self.t = tuner

    def build(self) -> dict:
        t = self.t
        aux_fitting_net = (
            t.fitting_net_params
            if getattr(t, "fitting_net_params", None)
            else {"type": "ener"}
        )
        # DOWNSTREAM branch: ener (legacy, sensitivity-analysis callers),
        # property (paper-faithful BOOM eval), or group_property (grouped/
        # assembly targets, e.g. OER overpotential). Default 'ener' for
        # back-compat with FakeTuners and existing callers that don't set
        # the attr.
        downstream_task_type = getattr(t, "downstream_task_type", "ener")
        # Both property and group_property get a fresh, RANDOM-initialized
        # downstream head sized by property_name/task_dim and follow the
        # qm9_gap paper-alignment defaults below; only legacy ener mode
        # reuses the aux branch's own fitting_net/finetune_head.
        is_random_downstream = downstream_task_type in _PROPERTY_LIKE_DOWNSTREAM_TYPES
        # Branch key for the downstream head. Paper qm9_gap/mft uses the task
        # type itself ("property" / "group_property"); legacy ener mode keeps
        # "DOWNSTREAM" so mp_data sensitivity-analysis configs stay
        # byte-for-byte unchanged (renaming would break the branch name in
        # their already-trained ckpts).
        downstream_key = downstream_task_type if is_random_downstream else "DOWNSTREAM"
        if downstream_task_type == "property":
            downstream_fitting_net = _build_property_fitting_net(t)
            downstream_loss = _build_property_loss()
        elif downstream_task_type == "group_property":
            downstream_fitting_net = _build_group_property_fitting_net(t)
            downstream_loss = _build_group_property_loss()
        else:
            downstream_fitting_net = aux_fitting_net
            downstream_loss = dict(_ENER_LOSS)

        # Paper qm9_gap/mft alignment is applied to both property and
        # group_property downstream modes. The legacy ener path (mp_data
        # sensitivity analysis) stays byte-for-byte unchanged.
        descriptor = {
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
            "activation_function": "silut:3.0"
            if is_random_downstream
            else "custom_silu:3.0",
            "precision": "float32",
            "use_tebd_bias": False,
            "concat_output_tebd": False,
            "exclude_types": [],
            "env_protection": 0.0,
            "trainable": True,
            "use_econf_tebd": False,
        }
        if is_random_downstream:
            descriptor["repflow"]["fix_stat_std"] = 0.3

        # MFT branch heads. In property/group_property mode the paper pins
        # finetune_head: the aux head loads from its named branch, the
        # downstream head is RANDOM-initialized (paper Eq 12). Legacy ener
        # mode keeps the original layout (no finetune_head on aux; downstream
        # = aux branch), including key order, so the emitted JSON is
        # byte-for-byte unchanged.
        if is_random_downstream:
            aux_head = {
                "type_map": "type_map",
                "descriptor": "dpa3_descriptor",
                "fitting_net": aux_fitting_net,
                "finetune_head": t.aux_branch,
            }
            downstream_head = {
                "finetune_head": "RANDOM",
                "type_map": "type_map",
                "descriptor": "dpa3_descriptor",
                "fitting_net": downstream_fitting_net,
            }
        else:
            aux_head = {
                "type_map": "type_map",
                "descriptor": "dpa3_descriptor",
                "fitting_net": aux_fitting_net,
            }
            downstream_head = {
                "finetune_head": t.aux_branch,
                "type_map": "type_map",
                "descriptor": "dpa3_descriptor",
                "fitting_net": downstream_fitting_net,
            }

        decay_steps = (
            t.decay_steps
            if getattr(t, "decay_steps", None) is not None
            else (1000 if is_random_downstream else 5000)
        )
        # Per-branch batch sizes: explicit override wins, then paper defaults
        # for property/group_property mode, then the single batch_size for
        # legacy ener mode.
        aux_batch = getattr(t, "aux_batch_size", None) or (
            "auto:128" if is_random_downstream else t.batch_size
        )
        downstream_batch = getattr(t, "downstream_batch_size", None) or (
            "auto:512" if is_random_downstream else t.batch_size
        )
        # Paper default 0.5/0.5; aux_prob (default 0.5) controls the split, the
        # downstream share is the complement. Legacy keeps downstream at 1.0.
        aux_prob = float(t.aux_prob)
        if not 0.0 <= aux_prob <= 1.0:
            raise ValueError(f"aux_prob must be in [0, 1]; got {t.aux_prob!r}.")
        downstream_prob = (1.0 - aux_prob) if is_random_downstream else 1.0

        aux_systems = t.aux_data if isinstance(t.aux_data, list) else [t.aux_data]
        train_systems = (
            t.train_data if isinstance(t.train_data, list) else [t.train_data]
        )
        valid_systems = None
        if getattr(t, "valid_data", None) is not None:
            valid_systems = (
                t.valid_data if isinstance(t.valid_data, list) else [t.valid_data]
            )

        training = {
            "model_prob": {t.aux_branch: aux_prob, downstream_key: downstream_prob},
            "data_dict": {
                t.aux_branch: {
                    "training_data": {"systems": aux_systems, "batch_size": aux_batch}
                },
                downstream_key: {
                    "training_data": {
                        "systems": train_systems,
                        "batch_size": downstream_batch,
                    }
                },
            },
            "numb_steps": t.max_steps,
            "save_freq": t.save_freq,
            # Pin the checkpoint prefix under output_dir (matching DPATrainer),
            # so DeePMD writes model.ckpt-*.pt there regardless of the process
            # cwd. Otherwise _freeze_ckpt()/evaluate()/predict() — which look
            # under output_dir — cannot find the checkpoint after a successful
            # fit() launched from another directory.
            "save_ckpt": os.path.join(t.output_dir, "model.ckpt"),
            "disp_freq": t.disp_freq,
            "seed": t.seed,
        }
        if valid_systems is not None:
            training["data_dict"][downstream_key]["validation_data"] = {
                "systems": valid_systems,
                "batch_size": downstream_batch,
            }

        if is_random_downstream:
            # Paper qm9_gap: gradient clipping at 5.0.
            training["gradient_max_norm"] = 5.0

        return {
            "model": {
                "shared_dict": {
                    "dpa3_descriptor": descriptor,
                    "type_map": t.type_map,
                },
                "model_dict": {t.aux_branch: aux_head, downstream_key: downstream_head},
            },
            "learning_rate": {
                "type": "exp",
                "start_lr": t.learning_rate,
                "stop_lr": t.stop_lr,
                "decay_steps": decay_steps,
                **(
                    {"warmup_steps": t.warmup_steps}
                    if getattr(t, "warmup_steps", 0) > 0
                    else {}
                ),
            },
            "loss_dict": {
                t.aux_branch: dict(_ENER_LOSS),
                downstream_key: downstream_loss,
            },
            "training": training,
        }

    def save(self, config: dict, path: str) -> str:
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return path

    def build_cmd(self, input_json_path: str) -> list[str]:
        t = self.t
        # MFT mode: do not pass --model-branch (branches are keyed by model_dict).
        # The full descriptor config is already in the JSON, so
        # --use-pretrain-script is not needed.
        return [
            resolve_dp_command(),
            "--pt",
            "train",
            input_json_path,
            "--skip-neighbor-stat",
            "--finetune",
            t.pretrained,
        ]
