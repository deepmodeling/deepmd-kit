# SPDX-License-Identifier: LGPL-3.0-or-later
import json

# Default property-head architecture for MFT DOWNSTREAM when
# downstream_task_type="property". Mirrors DPATrainer.DEFAULT_FITTING_NET
# (trainer.py L64-70) plus dim_case_embd=31, which the DPA-3.1-3M ckpt
# requires for the case-embedding layer in multi-task mode. (DPATrainer is
# single-task and doesn't need this field; in MFT the descriptor is shared
# across branches so the property head must declare it.)
_PROPERTY_FITTING_NET_BASE = {
    "type": "property",
    "neuron": [240, 240, 240],
    "activation_function": "tanh",
    "resnet_dt": True,
    "precision": "float32",
    "dim_case_embd": 31,
}


def _build_property_fitting_net(t) -> dict:
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
    def __init__(self, tuner):
        self.t = tuner

    def build(self) -> dict:
        t = self.t
        aux_fitting_net = (
            t.fitting_net_params
            if getattr(t, "fitting_net_params", None)
            else {"type": "ener"}
        )
        # DOWNSTREAM branch: ener (legacy, sensitivity-analysis callers) or
        # property (paper-faithful BOOM eval). Default 'ener' for back-compat
        # with FakeTuners and existing callers that don't set the attr.
        downstream_task_type = getattr(t, "downstream_task_type", "ener")
        is_property = downstream_task_type == "property"
        # Branch key for the downstream head. Paper qm9_gap/mft uses "property";
        # legacy ener mode keeps "DOWNSTREAM" so mp_data sensitivity-analysis
        # configs stay byte-for-byte unchanged (renaming would break the branch
        # name in their already-trained ckpts).
        downstream_key = "property" if is_property else "DOWNSTREAM"
        if is_property:
            downstream_fitting_net = _build_property_fitting_net(t)
            downstream_loss = _build_property_loss()
        else:
            downstream_fitting_net = aux_fitting_net
            downstream_loss = dict(_ENER_LOSS)

        # Paper qm9_gap/mft alignment is applied ONLY in property mode. The
        # legacy ener path (mp_data sensitivity analysis) stays byte-for-byte
        # unchanged.
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
            "activation_function": "silut:3.0" if is_property else "custom_silu:3.0",
            "precision": "float32",
            "use_tebd_bias": False,
            "concat_output_tebd": False,
            "exclude_types": [],
            "env_protection": 0.0,
            "trainable": True,
            "use_econf_tebd": False,
        }
        if is_property:
            descriptor["repflow"]["fix_stat_std"] = 0.3

        # MFT branch heads. In property mode the paper pins finetune_head:
        # the aux head loads from its named branch, the downstream property
        # head is RANDOM-initialized (paper Eq 12). Legacy ener mode keeps the
        # original layout (no finetune_head on aux; downstream = aux branch),
        # including key order, so the emitted JSON is byte-for-byte unchanged.
        if is_property:
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
            else (1000 if is_property else 5000)
        )
        # Per-branch batch sizes: explicit override wins, then paper defaults
        # for property mode, then the single batch_size for legacy ener mode.
        aux_batch = getattr(t, "aux_batch_size", None) or (
            "auto:128" if is_property else t.batch_size
        )
        downstream_batch = getattr(t, "downstream_batch_size", None) or (
            "auto:512" if is_property else t.batch_size
        )
        # Paper default 0.5/0.5; aux_prob (default 0.5) controls the split, the
        # downstream share is the complement. Legacy keeps downstream at 1.0.
        downstream_prob = (1.0 - t.aux_prob) if is_property else 1.0

        aux_systems = t.aux_data if isinstance(t.aux_data, list) else [t.aux_data]
        train_systems = (
            t.train_data if isinstance(t.train_data, list) else [t.train_data]
        )

        training = {
            "model_prob": {t.aux_branch: t.aux_prob, downstream_key: downstream_prob},
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
            "disp_freq": t.disp_freq,
            "seed": t.seed,
        }
        if is_property:
            # Paper qm9_gap: gradient clipping at 5.0.
            training["gradient_max_norm"] = 5.0

        return {
            "model": {
                "shared_dict": {
                    "dpa3_descriptor": descriptor,
                    "type_map": t.aux_type_map,
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

    def build_cmd(self, input_json_path: str) -> str:
        t = self.t
        # MFT 模式：不加 --model-branch（branch 由 model_dict key 控制）
        # descriptor 完整参数已在 config 中，不再需要 --use-pretrain-script
        return (
            f"dp --pt train {input_json_path} "
            f"--skip-neighbor-stat "
            f"--finetune {t.pretrained}"
        )
