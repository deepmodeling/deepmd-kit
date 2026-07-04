# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for TensorFlow 2 training internals."""

import importlib
from types import (
    SimpleNamespace,
)
from typing import (
    Any,
    ClassVar,
)

import numpy as np
import pytest

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
    TrainEntrypointOptions,
    TrainingTask,
)
from deepmd.tf2.common import (
    to_tf_tensor,
    wrap_tensor,
)
from deepmd.tf2.entrypoints.train import (
    TF2TrainEntrypoint,
)
from deepmd.tf2.env import (
    tf,
)
from deepmd.tf2.model.base_model import (
    forward_common_atomic,
)
from deepmd.tf2.train.trainer import (
    Trainer,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__init__ missing .*:DeprecationWarning:gast\\.astn"
)


class _LinearModel(tf.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = tf.Variable(2.0, dtype=tf.float64)

    def call(
        self,
        coord: Any,
        atype: Any,
        *,
        box: Any = None,
        fparam: Any = None,
        aparam: Any = None,
        charge_spin: Any = None,
    ) -> dict[str, Any]:
        del atype, box, fparam, aparam, charge_spin
        return {"prediction": coord * self.weight}


class _SquaredLoss:
    label_requirement: ClassVar[list[Any]] = []

    def __call__(
        self,
        *,
        learning_rate: Any,
        natoms: Any,
        model_dict: dict[str, Any],
        label_dict: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        del learning_rate
        diff = model_dict["prediction"] - label_dict["target"]
        loss = tf.reduce_mean(tf.square(diff))
        return loss, {
            "rmse": tf.sqrt(loss),
            "natoms": natoms,
            "l2_regularization": tf.constant(100.0, dtype=tf.float64),
        }


class _CountingAtomicModel:
    def __init__(self) -> None:
        self.calls = 0

    def forward_common_atomic(
        self,
        extended_coord: Any,
        extended_atype: Any,
        nlist: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del extended_atype, nlist, kwargs
        self.calls += 1
        coord = to_tf_tensor(extended_coord)
        return {"energy": wrap_tensor(tf.reduce_sum(coord * coord, axis=-1)[..., None])}


class _FakeEnergyModel:
    def __init__(self) -> None:
        self.atomic_model = _CountingAtomicModel()

    def atomic_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=True,
                    r_differentiable=True,
                )
            ]
        )


def _make_minimal_trainer() -> tuple[Trainer, _LinearModel]:
    trainer = object.__new__(Trainer)
    model = _LinearModel()
    trainer.models = {DEFAULT_TASK_KEY: model}
    trainer.losses = {DEFAULT_TASK_KEY: _SquaredLoss()}
    trainer.optimizer = tf.keras.optimizers.SGD(learning_rate=0.0)
    build = getattr(trainer.optimizer, "build", None)
    if callable(build):
        build(model.trainable_variables)
    trainer.gradient_max_norm = 0.0
    trainer.step = tf.Variable(0, dtype=tf.int64, trainable=False)
    trainer._compiled_train_steps = {}
    trainer._compiled_eval_steps = {}
    return trainer, model


def test_forward_common_atomic_reuses_taped_atomic_forward() -> None:
    model = _FakeEnergyModel()
    coord = tf.constant(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=tf.float64,
    )

    result = forward_common_atomic(
        model,
        wrap_tensor(coord),
        tf.constant([[0, 1]], dtype=tf.int32),
        tf.constant([[[0], [1]]], dtype=tf.int32),
    )

    assert model.atomic_model.calls == 1
    np.testing.assert_allclose(
        to_tf_tensor(result["energy_redu"]).numpy(),
        [[91.0]],
    )
    np.testing.assert_allclose(
        to_tf_tensor(result["energy_derv_r"]).numpy(),
        (-2.0 * coord[:, :, tf.newaxis, :]).numpy(),
    )


def test_compiled_train_step_is_tf_function_and_updates_model() -> None:
    trainer, model = _make_minimal_trainer()
    compiled = trainer._make_compiled_train_step(DEFAULT_TASK_KEY)

    assert hasattr(compiled, "get_concrete_function")
    more_loss = compiled(
        {
            "coord": tf.constant([[1.0]], dtype=tf.float64),
            "atype": tf.constant([[0]], dtype=tf.int32),
        },
        {"target": tf.constant([[0.0]], dtype=tf.float64)},
        tf.constant(1.0, dtype=tf.float64),
        tf.constant(0.1, dtype=tf.float64),
        tf.constant(1, dtype=tf.int64),
    )

    np.testing.assert_allclose(model.weight.numpy(), 1.6)
    assert int(trainer.step.numpy()) == 1
    assert more_loss["rmse"].numpy() == 2.0


def test_compiled_eval_step_returns_python_floats_without_l2_terms() -> None:
    trainer, model = _make_minimal_trainer()

    result = trainer._compiled_eval_step(
        DEFAULT_TASK_KEY,
        {
            "coord": tf.constant([[2.0]], dtype=tf.float64),
            "atype": tf.constant([[0]], dtype=tf.int32),
        },
        {"target": tf.constant([[1.0]], dtype=tf.float64)},
        tf.constant(3.0, dtype=tf.float64),
        tf.constant(0.1, dtype=tf.float64),
    )

    np.testing.assert_allclose(model.weight.numpy(), 2.0)
    assert result == {"rmse": 3.0, "natoms": 3.0}


def test_compiled_train_step_is_cached_per_task() -> None:
    trainer = object.__new__(Trainer)
    trainer._compiled_train_steps = {}
    calls: list[str] = []

    def make_compiled_step(task_key: str) -> Any:
        calls.append(task_key)

        def compiled_step(*args: Any) -> dict[str, Any]:
            del args
            return {"task": task_key}

        return compiled_step

    trainer._make_compiled_train_step = make_compiled_step

    assert trainer._compiled_train_step("a", {}, {}, 1.0, 0.1, 1)["task"] == "a"
    assert trainer._compiled_train_step("a", {}, {}, 1.0, 0.1, 2)["task"] == "a"
    assert trainer._compiled_train_step("b", {}, {}, 1.0, 0.1, 1)["task"] == "b"
    assert calls == ["a", "b"]


def test_train_step_passes_float_natoms_to_compiled_step() -> None:
    trainer = object.__new__(Trainer)
    trainer.lr_schedule = SimpleNamespace(value=lambda step: 0.25)
    trainer.get_data = lambda *, is_train, task_key: ({}, {}, 7)
    trainer._write_tensorboard_step = lambda *args, **kwargs: None
    captured: dict[str, Any] = {}

    def compiled_train_step(
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        natoms: Any,
        cur_lr: Any,
        next_step: Any,
    ) -> dict[str, Any]:
        del task_key, input_dict, label_dict
        captured["natoms"] = natoms
        captured["cur_lr"] = cur_lr
        captured["next_step"] = next_step
        return {"rmse": tf.constant(1.0, dtype=tf.float64)}

    trainer._compiled_train_step = compiled_train_step

    result = Trainer.train_step(
        trainer,
        TrainingTask(DEFAULT_TASK_KEY, SimpleNamespace()),
        4,
    )

    assert result.payload["cur_lr"] == 0.25
    assert captured["natoms"].dtype == tf.float64
    assert captured["natoms"].numpy() == 7.0
    assert captured["cur_lr"].dtype == tf.float64
    assert captured["next_step"].dtype == tf.int64
    assert captured["next_step"].numpy() == 5


def test_tensorboard_step_writes_tensors_without_float_sync(
    tmp_path: Any,
) -> None:
    trainer = object.__new__(Trainer)
    trainer.summary_writer = tf.summary.create_file_writer(str(tmp_path))
    trainer.tensorboard_freq = 1
    trainer.multi_task = False

    def fail_if_float_sync_is_used(more_loss: dict[str, Any]) -> dict[str, float]:
        del more_loss
        raise AssertionError("tensorboard path should not convert tensors to floats")

    trainer._more_loss_to_float = fail_if_float_sync_is_used

    Trainer._write_tensorboard_step(
        trainer,
        DEFAULT_TASK_KEY,
        display_step=1,
        learning_rate=0.1,
        more_loss={
            "rmse": tf.constant(1.0, dtype=tf.float64),
            "l2_regularization": tf.constant(2.0, dtype=tf.float64),
        },
    )
    trainer.summary_writer.close()


def test_train_entrypoint_builds_data_without_descriptor_rcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, Any], Any, Any, Any]] = []
    trainer_calls: list[dict[str, Any]] = []

    class FakeData:
        type_map: ClassVar[list[str]] = ["O", "H"]

        def print_summary(self, *args: Any) -> None:
            del args

    def fake_get_data(
        params: dict[str, Any],
        rcut: Any,
        type_map: Any,
        optional_type_map: Any,
    ) -> FakeData:
        calls.append((params, rcut, type_map, optional_type_map))
        return FakeData()

    class FakeTrainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            trainer_calls.append({"args": args, "kwargs": kwargs})

        def run(self) -> None:
            trainer_calls[-1]["ran"] = True

    module = importlib.import_module(TF2TrainEntrypoint.__module__)
    monkeypatch.setattr(module, "get_data", fake_get_data)
    monkeypatch.setattr(module, "DPTrainer", FakeTrainer)

    config = {
        "model": {"type_map": ["O", "H"]},
        "training": {
            "training_data": {"systems": ["train"]},
            "validation_data": {"systems": ["valid"]},
            "numb_steps": 1,
        },
    }

    TF2TrainEntrypoint().run_training(
        config,
        TrainEntrypointOptions(input_file="input.json"),
        neighbor_stat=0.5,
    )

    assert calls == [
        ({"systems": ["train"]}, None, ["O", "H"], None),
        ({"systems": ["valid"]}, None, ["O", "H"], None),
    ]
    assert trainer_calls[-1]["ran"] is True
    assert trainer_calls[-1]["kwargs"]["min_nbor_dist"] == 0.5
