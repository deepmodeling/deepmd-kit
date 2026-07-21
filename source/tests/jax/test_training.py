# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end tests for the local JAX training entrypoint."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from unittest.mock import (
    patch,
)

import numpy as np
import optax

from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
)
from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
    RankContext,
    TrainEntrypointOptions,
)
from deepmd.jax.entrypoints.freeze import (
    freeze,
)
from deepmd.jax.entrypoints.main import (
    main,
)
from deepmd.jax.entrypoints.train import (
    JAXTrainEntrypoint,
    update_sel,
)
from deepmd.jax.env import (
    jnp,
    nnx,
)
from deepmd.jax.infer.deep_eval import (
    DeepEval,
)
from deepmd.jax.model.hlo import (
    HLO,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
    _copy_matching_state_tree,
    _merge_descriptor_stats,
    _merge_fitting_param_stats,
    _scale_by_global_learning_rate,
)
from deepmd.jax.utils.finetune import (
    _load_model_params,
)
from deepmd.jax.utils.serialization import (
    _normalize_restored_state_keys,
)
from deepmd.utils.compat import (
    convert_optimizer_v31_to_v32,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)

MODEL_SE_E2_A = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [6, 12, 1],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [2, 4, 8],
        "resnet_dt": False,
        "axis_neuron": 2,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [4, 4, 4],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 1,
}


TRAINING_SCRIPT = """
from pathlib import Path
from unittest.mock import patch

from deepmd.main import main

with patch("deepmd.jax.entrypoints.train.SummaryPrinter.__call__"):
    main(["--jax", "train", "input.json", "--log-level", "2"])

for path in ["out.json", "lcurve.out", "checkpoint", "model-1.jax"]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
"""


_LCURVE_STEP_RE = re.compile(r"^\s*(\d+)\b")


def _lcurve_steps(path: Path) -> set[int]:
    """Return integer step numbers written in an lcurve.out file."""
    steps: set[int] = set()
    for line in path.read_text().splitlines():
        match = _LCURVE_STEP_RE.match(line)
        if match:
            steps.add(int(match.group(1)))
    return steps


def test_jax_optimizer_scales_updates_with_explicit_global_lr() -> None:
    """The optimizer LR comes from the loop step, not the per-task optax count."""
    tx = optax.chain(optax.scale_by_adam(), _scale_by_global_learning_rate())
    params = {"w": jnp.asarray(1.0)}
    grads = {"w": jnp.asarray(1.0)}
    state = tx.init(params)

    updates, state = tx.update(
        grads,
        state,
        params,
        learning_rate=jnp.asarray(0.2),
    )
    np.testing.assert_allclose(np.asarray(updates["w"]), -0.2, rtol=1e-5)

    updates, _ = tx.update(
        grads,
        state,
        params,
        learning_rate=jnp.asarray(0.05),
    )
    np.testing.assert_allclose(np.asarray(updates["w"]), -0.05, rtol=1e-5)


@patch("deepmd.jax.utils.finetune.serialize_from_file")
def test_jax_finetune_load_model_params_accepts_loader_paths(
    serialize_from_file,
) -> None:
    """Fine-tuning accepts every checkpoint path handled by serialize_from_file."""
    model_params = {"type_map": ["O"], "descriptor": {}, "fitting_net": {}}
    serialize_from_file.return_value = {"model_def_script": model_params}

    assert _load_model_params("checkpoint") == model_params
    serialize_from_file.assert_called_once_with("checkpoint")


def _minimal_jax_config(model_params: dict) -> dict:
    return {
        "model": model_params,
        "training": {
            "numb_steps": 1,
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": 0.001,
            "stop_lr": 1e-8,
            "decay_steps": 1,
        },
        "loss": {},
    }


def _minimal_jax_multitask_config(model_params: dict) -> dict:
    return {
        "model": model_params,
        "training": {
            "numb_steps": 1,
            "data_dict": {
                "task_a": {"training_data": {}},
                "task_b": {"training_data": {}},
            },
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": 0.001,
            "stop_lr": 1e-8,
            "decay_steps": 1,
        },
        "loss_dict": {
            "task_a": {},
            "task_b": {},
        },
    }


class _RequirementModel:
    """Minimal model double for trainer requirement-driven output tests."""

    def __init__(self) -> None:
        self.hessian_enable_calls = 0

    def enable_hessian(self) -> None:
        self.hessian_enable_calls += 1

    def get_dim_fparam(self) -> int:
        return 0


@patch("deepmd.jax.train.trainer.DPTrainer._build_losses")
@patch("deepmd.jax.train.trainer.get_model")
def test_jax_hessian_mode_follows_loss_data_requirement(
    get_model,
    build_losses,
) -> None:
    """The trainer consumes the loss requirement instead of its prefactors."""
    model = _RequirementModel()
    get_model.return_value = model
    build_losses.return_value = {
        "Default": SimpleNamespace(
            label_requirement=[DataRequirementItem("hessian", ndof=1)]
        )
    }
    model_params = {"type_map": ["O"], "descriptor": {}}

    trainer = DPTrainer(_minimal_jax_config(model_params))

    assert model.hessian_enable_calls == 1
    assert trainer.model_def_script["hessian_mode"] is True


@patch("deepmd.jax.train.trainer.DPTrainer._build_losses")
@patch("deepmd.jax.train.trainer.get_model")
def test_jax_multitask_hessian_only_enables_requesting_branch(
    get_model,
    build_losses,
) -> None:
    """Each multi-task branch follows its own loss data requirements."""
    model_a = _RequirementModel()
    model_b = _RequirementModel()
    get_model.side_effect = [model_a, model_b]
    build_losses.return_value = {
        "task_a": SimpleNamespace(
            label_requirement=[DataRequirementItem("hessian", ndof=1)]
        ),
        "task_b": SimpleNamespace(
            label_requirement=[DataRequirementItem("energy", ndof=1)]
        ),
    }
    model_params = {
        "model_dict": {
            "task_a": {"type_map": ["O"], "descriptor": {}},
            "task_b": {"type_map": ["O"], "descriptor": {}},
        }
    }

    trainer = DPTrainer(_minimal_jax_multitask_config(model_params))

    assert model_a.hessian_enable_calls == 1
    assert model_b.hessian_enable_calls == 0
    assert trainer.model_def_script["model_dict"]["task_a"]["hessian_mode"] is True
    assert "hessian_mode" not in trainer.model_def_script["model_dict"]["task_b"]


def _shared_jax_model_config(*, share_fitting: bool = True) -> dict:
    shared_dict: dict = {
        "shared_type_map": ["O", "H", "B"],
        "shared_descriptor": deepcopy(MODEL_SE_E2_A["descriptor"]),
    }
    fitting_ref_a: dict | str = deepcopy(MODEL_SE_E2_A["fitting_net"])
    fitting_ref_b: dict | str = deepcopy(MODEL_SE_E2_A["fitting_net"])
    if share_fitting:
        shared_dict["shared_fitting"] = deepcopy(MODEL_SE_E2_A["fitting_net"])
        fitting_ref_a = "shared_fitting"
        fitting_ref_b = "shared_fitting"
    return {
        "shared_dict": shared_dict,
        "model_dict": {
            "task_a": {
                "type_map": "shared_type_map",
                "descriptor": "shared_descriptor",
                "fitting_net": fitting_ref_a,
                "data_stat_nbatch": 1,
            },
            "task_b": {
                "type_map": "shared_type_map",
                "descriptor": "shared_descriptor",
                "fitting_net": fitting_ref_b,
                "data_stat_nbatch": 1,
            },
        },
    }


@patch("deepmd.jax.train.trainer.DPTrainer._build_losses")
@patch("deepmd.jax.train.trainer.DPTrainer._deserialize_models")
@patch("deepmd.jax.train.trainer.serialize_from_file")
def test_jax_init_model_preserves_input_model_script(
    serialize_from_file,
    deserialize_models,
    build_losses,
) -> None:
    """init_model loads weights without replacing input model metadata."""
    input_model = {"type_map": ["O"], "descriptor": {"input": True}}
    checkpoint_model = {"type_map": ["O"], "descriptor": {"checkpoint": True}}
    serialize_from_file.return_value = {
        "model": {},
        "model_def_script": checkpoint_model,
    }
    deserialize_models.return_value = {
        "Default": SimpleNamespace(get_dim_fparam=lambda: 0)
    }
    build_losses.return_value = {"Default": SimpleNamespace(label_requirement=[])}

    trainer = DPTrainer(_minimal_jax_config(input_model), init_model="model-1.jax")

    assert trainer.model_def_script == input_model
    assert trainer.model_params_by_task["Default"] == input_model


@patch("deepmd.jax.train.trainer.DPTrainer._build_losses")
@patch("deepmd.jax.train.trainer.DPTrainer._deserialize_models")
@patch("deepmd.jax.train.trainer.serialize_from_file")
def test_jax_restart_uses_checkpoint_model_script(
    serialize_from_file,
    deserialize_models,
    build_losses,
) -> None:
    """Restart keeps checkpoint metadata and resumed current_step."""
    input_model = {"type_map": ["O"], "descriptor": {"input": True}}
    checkpoint_model = {
        "type_map": ["O"],
        "descriptor": {"checkpoint": True},
        "current_step": 7,
    }
    serialize_from_file.return_value = {
        "model": {},
        "model_def_script": checkpoint_model,
    }
    deserialize_models.return_value = {
        "Default": SimpleNamespace(get_dim_fparam=lambda: 0)
    }
    build_losses.return_value = {"Default": SimpleNamespace(label_requirement=[])}

    trainer = DPTrainer(_minimal_jax_config(input_model), restart="model-7.jax")

    assert trainer.model_def_script == checkpoint_model
    assert trainer.model_params_by_task["Default"] == checkpoint_model
    assert trainer.start_step == 7


def test_jax_train_entrypoint_preprocesses_shared_dict() -> None:
    """JAX multi-task preprocessing expands shared_dict references."""
    entrypoint = JAXTrainEntrypoint()
    config = {
        "model": _shared_jax_model_config(),
        "training": {},
    }

    updated = entrypoint.preprocess_config(
        config,
        TrainEntrypointOptions(input_file="input.json"),
    )

    model_dict = updated["model"]["model_dict"]
    assert model_dict["task_a"]["type_map"] == ["O", "H", "B"]
    assert model_dict["task_b"]["descriptor"]["type"] == "se_e2_a"
    assert entrypoint.shared_links is not None
    assert set(entrypoint.shared_links) == {"shared_descriptor", "shared_fitting"}


def test_jax_train_entrypoint_keeps_multitask_without_shared_dict() -> None:
    """JAX multi-task configs without shared_dict keep the existing path."""
    entrypoint = JAXTrainEntrypoint()
    config = {
        "model": {
            "model_dict": {
                "task_a": deepcopy(MODEL_SE_E2_A),
                "task_b": deepcopy(MODEL_SE_E2_A),
            },
        },
        "training": {},
    }

    updated = entrypoint.preprocess_config(
        config,
        TrainEntrypointOptions(input_file="input.json"),
    )

    assert updated["model"]["model_dict"]["task_a"]["type_map"] == ["O", "H", "B"]
    assert entrypoint.shared_links is None


@patch("deepmd.jax.entrypoints.train.serialize_from_file")
def test_jax_train_entrypoint_preprocesses_shared_dict_after_model_replacement(
    serialize_from_file,
) -> None:
    """JAX shared links match the final model after pretrain-script replacement."""
    input_model = {
        "shared_dict": {
            "input_type_map": ["input"],
            "input_descriptor": deepcopy(MODEL_SE_E2_A["descriptor"]),
        },
        "model_dict": {
            "task_a": {
                "type_map": "input_type_map",
                "descriptor": "input_descriptor",
                "fitting_net": deepcopy(MODEL_SE_E2_A["fitting_net"]),
            },
            "task_b": {
                "type_map": "input_type_map",
                "descriptor": "input_descriptor",
                "fitting_net": deepcopy(MODEL_SE_E2_A["fitting_net"]),
            },
        },
    }
    checkpoint_model = {
        "shared_dict": {
            "checkpoint_type_map": ["O", "H", "B"],
            "checkpoint_descriptor": deepcopy(MODEL_SE_E2_A["descriptor"]),
        },
        "model_dict": {
            "task_a": {
                "type_map": "checkpoint_type_map",
                "descriptor": "checkpoint_descriptor",
                "fitting_net": deepcopy(MODEL_SE_E2_A["fitting_net"]),
            },
            "task_b": {
                "type_map": "checkpoint_type_map",
                "descriptor": "checkpoint_descriptor",
                "fitting_net": deepcopy(MODEL_SE_E2_A["fitting_net"]),
            },
        },
    }
    serialize_from_file.return_value = {
        "model_def_script": checkpoint_model,
    }
    entrypoint = JAXTrainEntrypoint()
    config = {"model": input_model, "training": {}}

    updated = entrypoint.preprocess_config(
        config,
        TrainEntrypointOptions(
            input_file="input.json",
            init_model="model.jax",
            use_pretrain_script=True,
        ),
    )

    assert updated["model"]["model_dict"]["task_a"]["type_map"] == ["O", "H", "B"]
    assert entrypoint.shared_links is not None
    assert set(entrypoint.shared_links) == {"checkpoint_descriptor"}
    assert "input_descriptor" not in entrypoint.shared_links


def test_jax_trainer_applies_shared_dict_links() -> None:
    """Trainer-level sharing links descriptor and fitting-net parameters."""
    trainer = DPTrainer(
        _minimal_jax_multitask_config(_shared_jax_model_config()),
    )

    trainer._share_model_params(resume=True)

    model_a = trainer.models["task_a"]
    model_b = trainer.models["task_b"]
    assert model_a.get_descriptor() is model_b.get_descriptor()
    assert model_a.get_fitting_net() is not model_b.get_fitting_net()
    assert model_a.get_fitting_net().nets is model_b.get_fitting_net().nets


class _FakeEnvMatStatSe:
    def __init__(self, descriptor) -> None:
        self.descriptor = descriptor
        self.stats = {}

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        stat = self.stats["env"]
        return (
            np.asarray([stat.compute_avg()], dtype=np.float64),
            np.asarray([stat.compute_std()], dtype=np.float64),
        )


class _DescriptorWithStats:
    def __init__(self, stats: dict[str, StatItem]) -> None:
        self.stats = stats
        self.davg = np.asarray([0.0], dtype=np.float64)
        self.dstd = np.asarray([1.0], dtype=np.float64)


def test_jax_shared_descriptor_stats_merge_weighted_values() -> None:
    """Shared descriptor merge recomputes weighted avg/std for nested stats."""
    base = _DescriptorWithStats({"env": StatItem(number=2, sum=4, squared_sum=10)})
    link = _DescriptorWithStats({"env": StatItem(number=4, sum=20, squared_sum=104)})
    base.se_atten = _DescriptorWithStats(
        {"env": StatItem(number=2, sum=6, squared_sum=18)}
    )
    link.se_atten = _DescriptorWithStats(
        {"env": StatItem(number=4, sum=28, squared_sum=200)}
    )

    with patch("deepmd.dpmodel.utils.env_mat_stat.EnvMatStatSe", _FakeEnvMatStatSe):
        _merge_descriptor_stats(base, link, model_prob=0.5)

    np.testing.assert_allclose(base.davg, [3.5])
    np.testing.assert_allclose(base.dstd, [np.sqrt(3.25)])
    np.testing.assert_allclose(base.se_atten.davg, [5.0])
    np.testing.assert_allclose(base.se_atten.dstd, [np.sqrt(4.5)])
    assert base.stats["env"].number == 4
    assert base.se_atten.stats["env"].number == 4


class _FittingWithStats:
    def __init__(self, param_stats: dict[str, list[StatItem]]) -> None:
        self.numb_fparam = len(param_stats.get("fparam", []))
        self.numb_aparam = len(param_stats.get("aparam", []))
        self.fparam_avg = jnp.asarray(np.zeros(self.numb_fparam))
        self.fparam_inv_std = jnp.asarray(np.ones(self.numb_fparam))
        self.aparam_avg = jnp.asarray(np.zeros(self.numb_aparam))
        self.aparam_inv_std = jnp.asarray(np.ones(self.numb_aparam))
        self._param_stats = param_stats

    def get_param_stats(self) -> dict[str, list[StatItem]]:
        return self._param_stats


def test_jax_shared_fitting_stats_merge_weighted_values() -> None:
    """Shared fitting merge recomputes avg and protected inverse std."""
    base = _FittingWithStats(
        {
            "fparam": [StatItem(number=2, sum=4, squared_sum=10)],
            "aparam": [StatItem(number=2, sum=4, squared_sum=8)],
        }
    )
    link = _FittingWithStats(
        {
            "fparam": [StatItem(number=4, sum=20, squared_sum=104)],
            "aparam": [StatItem(number=2, sum=4, squared_sum=8)],
        }
    )

    _merge_fitting_param_stats(
        base,
        link,
        model_prob=0.5,
        protection=0.25,
    )

    np.testing.assert_allclose(np.asarray(base.fparam_avg), [3.5])
    np.testing.assert_allclose(np.asarray(base.fparam_inv_std), [1.0 / np.sqrt(3.25)])
    np.testing.assert_allclose(np.asarray(base.aparam_avg), [2.0])
    np.testing.assert_allclose(np.asarray(base.aparam_inv_std), [4.0])
    assert base._param_stats["fparam"][0].number == 4
    assert base._param_stats["aparam"][0].number == 3


def test_jax_full_validator_saves_directory_best_checkpoint(tmp_path: Path) -> None:
    """JAX full validation uses .jax directory checkpoints."""
    from deepmd.jax.train.validation import (
        JAXFullValidator,
    )

    state_store: dict = {}
    validator = JAXFullValidator(
        validating_params={
            "full_validation": True,
            "validation_freq": 1,
            "save_best": True,
            "max_best_ckpt": 1,
            "validation_metric": "E:MAE",
            "full_val_file": str(tmp_path / "val.log"),
            "full_val_start": 0.0,
        },
        validation_data=SimpleNamespace(),
        model=SimpleNamespace(),
        state_store=state_store,
        num_steps=2,
        rank=0,
        restart_training=False,
        checkpoint_dir=tmp_path,
    )

    def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
        del lr, step
        path.mkdir(parents=True)

    with patch.object(
        validator,
        "evaluate_all_systems",
        return_value={"mae_e_per_atom": 1.0},
    ):
        result = validator.run(
            step_id=1,
            display_step=1,
            lr=0.001,
            save_checkpoint=save_checkpoint,
        )

    assert result is not None
    assert (tmp_path / "best.ckpt-1.t-1.jax").is_dir()
    assert state_store["full_validation_topk_records"] == [{"metric": 1.0, "step": 1}]
    assert "1000.0" in (tmp_path / "val.log").read_text()


def test_jax_full_validator_broadcasts_rank_zero_errors(tmp_path: Path) -> None:
    """JAX full validation synchronizes rank-0 failures to peer processes."""
    from deepmd.jax.train.validation import (
        JAXFullValidator,
    )

    validator = JAXFullValidator(
        validating_params={
            "full_validation": True,
            "validation_freq": 1,
            "save_best": False,
            "max_best_ckpt": 1,
            "validation_metric": "E:MAE",
            "full_val_file": str(tmp_path / "val.log"),
            "full_val_start": 0.0,
        },
        validation_data=SimpleNamespace(),
        model=SimpleNamespace(),
        state_store={},
        num_steps=2,
        rank=1,
        restart_training=False,
        checkpoint_dir=tmp_path,
    )

    with (
        patch("deepmd.jax.train.validation.jax.process_count", return_value=2),
        patch(
            "jax.experimental.multihost_utils.broadcast_one_to_all",
            return_value=np.asarray(True),
        ) as broadcast_one_to_all,
    ):
        assert (
            validator.propagate_error(None)
            == "Full validation failed on rank 0; see rank-0 logs."
        )

    broadcast_one_to_all.assert_called_once()
    assert broadcast_one_to_all.call_args.kwargs["is_source"] is False


def test_jax_full_validation_hook_uses_display_step() -> None:
    """JAX full-validation checkpoints carry one-based display steps."""
    calls: list[dict] = []
    save_calls: list[tuple[Path, float, int]] = []

    class FakeValidator:
        def run(self, **kwargs) -> None:
            calls.append(kwargs)
            kwargs["save_checkpoint"](Path("best.jax"), lr=kwargs["lr"], step=99)

    trainer = DPTrainer.__new__(DPTrainer)
    trainer.full_validator = FakeValidator()

    def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
        save_calls.append((path, lr, step))

    trainer._save_full_validation_checkpoint = save_checkpoint

    DPTrainer.run_full_validation(
        trainer,
        step=0,
        display_step=1,
        learning_rate=0.25,
    )

    assert calls[0]["step_id"] == 1
    assert calls[0]["display_step"] == 1
    assert calls[0]["lr"] == 0.25
    assert save_calls == [(Path("best.jax"), 0.25, 99)]


class _BiasModel(nnx.Module):
    def __init__(self, value: float) -> None:
        self.bias = nnx.Param(jnp.asarray([value]))


def _bias_sync_trainer(rank: int) -> DPTrainer:
    trainer = DPTrainer.__new__(DPTrainer)
    trainer.rank_context = RankContext(rank=rank, world_size=2)
    trainer.models = {DEFAULT_TASK_KEY: _BiasModel(0.0)}
    trainer._sample_funcs = {DEFAULT_TASK_KEY: object()}
    trainer.model_keys = [DEFAULT_TASK_KEY]
    return trainer


def test_jax_change_bias_after_training_broadcasts_chief_state() -> None:
    """Rank 0 recomputes post-training bias and broadcasts the resulting state."""
    trainer = _bias_sync_trainer(rank=0)

    def change_bias(models, *args, **kwargs) -> None:
        del args, kwargs
        nnx.update(models[DEFAULT_TASK_KEY], {"bias": jnp.asarray([3.0])})

    with (
        patch(
            "deepmd.jax.train.trainer.change_model_out_bias_by_task",
            side_effect=change_bias,
        ) as change_model_out_bias_by_task,
        patch(
            "jax.experimental.multihost_utils.broadcast_one_to_all",
            side_effect=lambda state, **kwargs: state,
        ) as broadcast_one_to_all,
    ):
        trainer._change_bias_after_training()

    change_model_out_bias_by_task.assert_called_once()
    broadcast_one_to_all.assert_called_once()
    assert broadcast_one_to_all.call_args.kwargs["is_source"] is True
    np.testing.assert_allclose(
        np.asarray(trainer.models[DEFAULT_TASK_KEY].bias.value),
        [3.0],
    )


def test_jax_change_bias_after_training_uses_broadcast_on_peer_rank() -> None:
    """Peer ranks receive rank-0 post-training bias instead of recomputing it."""
    trainer = _bias_sync_trainer(rank=1)

    with (
        patch(
            "deepmd.jax.train.trainer.change_model_out_bias_by_task",
        ) as change_model_out_bias_by_task,
        patch(
            "jax.experimental.multihost_utils.broadcast_one_to_all",
            return_value={"bias": jnp.asarray([5.0])},
        ) as broadcast_one_to_all,
    ):
        trainer._change_bias_after_training()

    change_model_out_bias_by_task.assert_not_called()
    broadcast_one_to_all.assert_called_once()
    assert broadcast_one_to_all.call_args.kwargs["is_source"] is False
    np.testing.assert_allclose(
        np.asarray(trainer.models[DEFAULT_TASK_KEY].bias.value),
        [5.0],
    )


class TestJAXTraining(unittest.TestCase):
    """Regression tests for complete JAX training runs."""

    def setUp(self) -> None:
        """Create a temporary work directory with a one-step training input."""
        self.work_dir = Path(tempfile.mkdtemp())
        self.cwd = Path.cwd()
        os.chdir(self.work_dir)

        source_dir = Path(__file__).resolve().parents[1] / "pt" / "water"
        shutil.copytree(source_dir, self.work_dir / "water")
        data_file = [str(self.work_dir / "water" / "data" / "single")]

        with (self.work_dir / "water" / "se_atten.json").open() as f:
            self.config = json.load(f)
        self.config = convert_optimizer_v31_to_v32(self.config, warning=False)
        self.config["model"] = MODEL_SE_E2_A
        self.config["model"]["data_stat_nbatch"] = 1
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["disp_freq"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["save_ckpt"] = "model"

        self.input_file = self.work_dir / "input.json"
        with self.input_file.open("w") as f:
            json.dump(self.config, f)

    def tearDown(self) -> None:
        """Remove temporary training outputs."""
        os.chdir(self.cwd)
        shutil.rmtree(self.work_dir)

    def test_train_entrypoint_runs_one_step_from_scratch(self) -> None:
        """Run local JAX training in a child process and check artifacts."""
        if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ):
            # TODO: Re-enable this in GitHub CUDA CI once the hosted/self-hosted
            # runner JAX/PJRT abort is understood. The same test passes on a
            # local GPU, but the GitHub Actions CUDA job can terminate with
            # CUDA_ERROR_LAUNCH_FAILED while PJRT releases device buffers.
            self.skipTest(
                "JAX training is temporarily skipped on GitHub Actions CUDA runners"
            )

        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(TRAINING_SCRIPT)],
            cwd=self.work_dir,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(1, _lcurve_steps(self.work_dir / "lcurve.out"))

    @patch("deepmd.jax.entrypoints.train.get_data")
    @patch("deepmd.jax.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel_uses_jax_neighbor_stat(self, get_nbor_stat, get_data) -> None:
        """JAX update_sel should calculate neighbor statistics instead of skipping."""
        get_nbor_stat.return_value = 0.5, [10, 20]
        jdata = {
            "model": {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "rcut": 6.0,
                    "sel": "auto",
                },
            },
            "training": {"training_data": {}},
        }

        updated, min_nbor_dist = update_sel(jdata)

        self.assertEqual(updated["model"]["descriptor"]["sel"], [12, 24])
        self.assertEqual(min_nbor_dist, 0.5)
        get_data.assert_called_once_with({}, 0, ["O", "H"], None)
        get_nbor_stat.assert_called_once()

    def test_train_entrypoint_rejects_remaining_unsupported_features(self) -> None:
        """JAX train gates features that are still backend-specific gaps."""
        entrypoint = JAXTrainEntrypoint()

        cases = [
            (
                {"model": {}, "training": {}},
                TrainEntrypointOptions(
                    input_file="input.json",
                    init_frz_model="frozen_model.pb",
                ),
                "init_frz_model",
            ),
        ]

        for config, options, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(NotImplementedError, message):
                    entrypoint.validate_options(config, options)

    @patch("deepmd.jax.utils.finetune.get_finetune_rules")
    def test_train_entrypoint_preprocesses_finetune_rules(
        self, get_finetune_rules
    ) -> None:
        """JAX train preprocesses fine-tuning config through backend rules."""
        get_finetune_rules.return_value = (
            {"type_map": ["O"], "descriptor": {}, "fitting_net": {}},
            {"Default": object()},
        )
        entrypoint = JAXTrainEntrypoint()
        config = {"model": {"type_map": ["O"]}, "training": {}}

        updated = entrypoint.preprocess_config(
            config,
            TrainEntrypointOptions(
                input_file="input.json",
                finetune="pretrain.jax",
                model_branch="head",
                use_pretrain_script=True,
            ),
        )

        self.assertEqual(updated["model"]["type_map"], ["O"])
        self.assertIsNotNone(entrypoint.finetune_links)
        get_finetune_rules.assert_called_once_with(
            "pretrain.jax",
            {"type_map": ["O"]},
            model_branch="head",
            change_model_params=True,
        )

    @patch("deepmd.jax.entrypoints.train.get_data")
    @patch("deepmd.jax.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel_supports_multitask(self, get_nbor_stat, get_data) -> None:
        """JAX update_sel updates each multi-task branch."""
        get_nbor_stat.return_value = 0.5, [10, 20]
        model_config = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "rcut": 6.0,
                "sel": "auto",
            },
        }
        jdata = {
            "model": {
                "model_dict": {
                    "task_a": json.loads(json.dumps(model_config)),
                    "task_b": json.loads(json.dumps(model_config)),
                }
            },
            "training": {
                "data_dict": {
                    "task_a": {"training_data": {"systems": ["a"]}},
                    "task_b": {"training_data": {"systems": ["b"]}},
                }
            },
        }

        updated, min_nbor_dist = update_sel(jdata, multi_task=True)

        self.assertEqual(
            updated["model"]["model_dict"]["task_a"]["descriptor"]["sel"], [12, 24]
        )
        self.assertEqual(
            updated["model"]["model_dict"]["task_b"]["descriptor"]["sel"], [12, 24]
        )
        self.assertEqual(
            min_nbor_dist,
            {"task_a": 0.5, "task_b": 0.5},
        )
        self.assertEqual(get_data.call_count, 2)
        self.assertEqual(get_nbor_stat.call_count, 2)

    @patch("deepmd.jax.entrypoints.freeze.deserialize_to_file")
    @patch("deepmd.jax.entrypoints.freeze.serialize_from_file")
    def test_freeze_entrypoint_uses_checkpoint_pointer(
        self, serialize_from_file, deserialize_to_file
    ) -> None:
        """Freeze resolves the stable checkpoint pointer and forwards Hessian."""
        checkpoint_dir = self.work_dir / "ckpt"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "checkpoint").write_text("model-1.jax")
        serialize_from_file.return_value = {"model": {}, "model_def_script": {}}

        freeze(
            checkpoint_folder=str(checkpoint_dir), output="frozen_model", hessian=True
        )

        serialize_from_file.assert_called_once_with(str(checkpoint_dir / "model-1.jax"))
        deserialize_to_file.assert_called_once_with(
            "frozen_model.hlo", serialize_from_file.return_value, hessian=True
        )

    @patch("deepmd.jax.entrypoints.main.freeze")
    def test_main_dispatches_freeze(self, freeze_entrypoint) -> None:
        """JAX CLI main imports and dispatches the freeze command."""
        args = argparse.Namespace(
            command="freeze",
            log_level=2,
            log_path=None,
            checkpoint_folder=".",
            output="frozen_model",
            hessian=False,
        )

        main(args)

        freeze_entrypoint.assert_called_once()
        self.assertIn("hessian", freeze_entrypoint.call_args.kwargs)
        self.assertFalse(freeze_entrypoint.call_args.kwargs["hessian"])

    def test_hlo_hessian_mode_updates_output_def(self) -> None:
        """HLO output definition should expose Hessian when requested."""
        hlo = object.__new__(HLO)
        hlo._model_output_type = ["energy"]
        hlo.model_def_script = json.dumps({"hessian_mode": True})

        output_def = hlo.model_output_def()

        self.assertTrue(output_def["energy"].r_hessian)
        self.assertIn("energy_derv_r_derv_r", output_def.keys())

    def test_deep_eval_requests_hessian_for_hessian_model(self) -> None:
        """Non-atomic JAX evaluation should request Hessian outputs."""
        hlo = object.__new__(HLO)
        hlo._model_output_type = ["energy"]
        hlo.model_def_script = json.dumps({"hessian_mode": True})
        deep_eval = object.__new__(DeepEval)
        deep_eval.output_def = hlo.model_output_def()
        deep_eval.dp = SimpleNamespace(
            get_model_def_script=lambda: json.dumps({"hessian_mode": True})
        )

        request_defs = deep_eval._get_request_defs(atomic=False)

        self.assertTrue(deep_eval.get_has_hessian())
        self.assertIn(
            OutputVariableCategory.DERV_R_DERV_R,
            {odef.category for odef in request_defs},
        )

    def test_deep_eval_skips_hessian_for_standard_model(self) -> None:
        """Standard JAX evaluation should not request Hessian outputs."""
        hlo = object.__new__(HLO)
        hlo._model_output_type = ["energy"]
        hlo.model_def_script = json.dumps({"hessian_mode": False})
        deep_eval = object.__new__(DeepEval)
        deep_eval.output_def = hlo.model_output_def()
        deep_eval.dp = SimpleNamespace(
            get_model_def_script=lambda: json.dumps({"hessian_mode": False})
        )

        request_defs = deep_eval._get_request_defs(atomic=False)

        self.assertFalse(deep_eval.get_has_hessian())
        self.assertNotIn(
            OutputVariableCategory.DERV_R_DERV_R,
            {odef.category for odef in request_defs},
        )


def test_jax_finetune_state_copy_preserves_random_fitting_target_leaves() -> None:
    """Random fitting should copy descriptor leaves only."""
    target = {
        "descriptor": {"w": np.zeros((2,), dtype=np.float64)},
        "fitting_net": {"w": np.zeros((2,), dtype=np.float64)},
        "output": {"bias": np.zeros((1,), dtype=np.float64)},
    }
    source = {
        "descriptor": {"w": np.ones((2,), dtype=np.float64)},
        "fitting_net": {"w": np.full((2,), 2.0, dtype=np.float64)},
        "output": {"bias": np.ones((1,), dtype=np.float64)},
    }

    copied = _copy_matching_state_tree(target, source, random_fitting=True)

    np.testing.assert_array_equal(copied["descriptor"]["w"], source["descriptor"]["w"])
    np.testing.assert_array_equal(
        copied["fitting_net"]["w"], target["fitting_net"]["w"]
    )
    np.testing.assert_array_equal(copied["output"]["bias"], target["output"]["bias"])


def test_jax_finetune_state_copy_requires_matching_leaf_shape() -> None:
    """Mismatched state leaves are left unchanged."""
    target = {"descriptor": {"w": np.zeros((2,), dtype=np.float64)}}
    source = {"descriptor": {"w": np.ones((3,), dtype=np.float64)}}

    copied = _copy_matching_state_tree(target, source, random_fitting=False)

    np.testing.assert_array_equal(copied["descriptor"]["w"], target["descriptor"]["w"])


def test_jax_multitask_state_key_normalization_preserves_numeric_task_names() -> None:
    """Numeric-looking task keys are branch names, not layer indices."""
    state = {
        "models": {
            "1": {"layers": {"0": {"w": 1}}},
            "task": {"layers": {"0": {"w": 2}}},
        }
    }
    model_def_script = {"model_dict": {"1": {}, "task": {}}}

    _normalize_restored_state_keys(state, model_def_script)

    assert "1" in state["models"]
    assert 1 not in state["models"]
    assert 0 in state["models"]["1"]["layers"]
    assert 0 in state["models"]["task"]["layers"]
