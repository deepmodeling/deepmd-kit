# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training, restart, and task-sharing contracts for JAX DPA4."""

from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import optax
import pytest

from deepmd.dpmodel.loss.property import (
    PropertyLoss,
)
from deepmd.dpmodel.train import (
    TrainingTask,
)
from deepmd.entrypoints.convert_backend import (
    convert_backend,
)
from deepmd.jax.common import (
    to_flat_state,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.jax.optimizer import (
    build_adam,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
    _clip_gradients,
    _global_gradient_norm,
    _task_parameter_masks,
)
from deepmd.jax.utils.finetune import (
    merged_initialization_source,
)
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)
from deepmd.utils.data_system import (
    get_data,
)

from .test_dpa4_model import (
    model_config,
)


def _dataset(path: Path) -> Path:
    """Write one deterministic frame with ordinary and magnetic labels."""
    path.mkdir()
    (path / "type.raw").write_text("0\n1\n0\n")
    (path / "type_map.raw").write_text("O\nH\n")
    (path / "nopbc").touch()
    data = path / "set.000"
    data.mkdir()
    np.save(data / "coord.npy", [[0.0, 0.0, 0.0, 1.1, 0.2, 0.3, 0.4, 1.3, 0.6]])
    np.save(data / "energy.npy", [[0.3]])
    np.save(data / "force.npy", np.arange(9).reshape(1, 9) * 0.01)
    np.save(data / "spin.npy", [[0.3, -0.2, 0.4, 0.0, 0.0, 0.0, 0.2, 0.4, -0.1]])
    np.save(data / "force_mag.npy", np.arange(9).reshape(1, 9) * 0.02)
    np.save(data / "charge_spin.npy", [[0.0, 1.0]])
    return path


def _config(path: Path, steps: int, *, spin: bool = False) -> dict[str, Any]:
    """Build a short run that exercises checkpoint and parameter trajectories."""
    model = model_config(spin=spin, bridging=spin)
    model["data_stat_nbatch"] = 1
    model["descriptor"]["sel"] = 1
    model["descriptor"]["add_chg_spin_ebd"] = True
    loss = {
        "type": "ener_spin" if spin else "ener",
        "start_pref_e": 1.0,
        "limit_pref_e": 1.0,
    }
    force_keys = ("fr", "fm") if spin else ("f",)
    for key in force_keys:
        loss[f"start_pref_{key}"] = 1.0
        loss[f"limit_pref_{key}"] = 1.0
    return {
        "model": model,
        "loss": loss,
        "optimizer": {"type": "HybridMuon" if spin else "AdamW"},
        "learning_rate": {
            "type": "exp",
            "start_lr": 1e-3,
            "stop_lr": 1e-3,
            "decay_steps": 1,
        },
        "training": {
            "numb_steps": steps,
            "disp_freq": 1,
            "save_freq": 2,
            "save_ckpt": str(path / "model"),
            "save_dir": str(path / "checkpoints"),
            "disp_file": str(path / "lcurve.out"),
            "gradient_max_norm": 5.0,
            "enable_ema": True,
            "ema_decay": 0.8,
            "disp_avg": True,
        },
    }


def _run(config: dict[str, Any], dataset: Path, **kwargs: Any) -> DPTrainer:
    trainer = DPTrainer(deepcopy(config), **kwargs)
    data = get_data(
        {"systems": [str(dataset)], "batch_size": 1},
        trainer.model.get_rcut(),
        trainer.model.get_type_map(),
        None,
    )
    trainer.train(data, data)
    return trainer


def _assert_states_equal(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(nnx.state(left))
    right_leaves, right_tree = jax.tree.flatten(nnx.state(right))
    assert left_tree == right_tree
    for expected, actual in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("random_gamma", [False, True])
def test_dpa4_restart_preserves_optimizer_and_ema(
    tmp_path: Path, random_gamma: bool
) -> None:
    """A resumed run follows the uninterrupted model and EMA trajectories."""
    dataset = _dataset(tmp_path / "data")
    continuous_config = _config(tmp_path / "continuous", 3)
    continuous_config["model"]["descriptor"]["random_gamma"] = random_gamma
    continuous = _run(continuous_config, dataset)
    partial_config = _config(tmp_path / "resumed", 2)
    partial_config["model"]["descriptor"]["random_gamma"] = random_gamma
    _run(partial_config, dataset)
    partial_config["training"]["numb_steps"] = 3
    resumed = _run(
        partial_config,
        dataset,
        restart=str(tmp_path / "resumed" / "model.jax"),
    )
    assert resumed.start_step == 2
    _assert_states_equal(continuous.models, resumed.models)
    _assert_states_equal(continuous.optimizer, resumed.optimizer)
    _assert_states_equal(continuous.ema_models, resumed.ema_models)
    assert (tmp_path / "resumed" / "checkpoints" / "model-3.jax").is_dir()
    assert (tmp_path / "resumed" / "model_ema.jax").is_symlink()


def test_spin_zbl_training_and_ema_full_validation(tmp_path: Path) -> None:
    """Magnetic-force supervision and both validators consume all conditions."""
    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 1, spin=True)
    config["validating"] = {
        "full_validation": True,
        "full_validation_ema": True,
        "full_val_start": 0.0,
        "validation_freq": 1,
        "validation_metric": "FM:RMSE",
        "full_val_file": str(tmp_path / "run" / "val.log"),
        "save_best_dir": str(tmp_path / "run" / "best"),
    }
    trainer = _run(config, dataset)
    assert trainer.full_validator.topk_records
    assert trainer.ema_full_validator.topk_records
    assert list((tmp_path / "run" / "best").glob("best.ckpt-*.jax"))
    assert list((tmp_path / "run" / "best").glob("best_ema.ckpt-*.jax"))
    curve = np.loadtxt(tmp_path / "run" / "lcurve.out")
    assert np.isfinite(curve).all()


def test_bridged_training_records_contacts_before_the_step(tmp_path: Path) -> None:
    """Training batches set the data edges that the model and its EMA read."""
    dataset = _dataset(tmp_path / "data")
    coordinates = np.load(dataset / "set.000" / "coord.npy")
    # An O-H contact below the covalent start of the pair enters the record.
    coordinates[0, 3:6] = [0.8, 0.0, 0.0]
    np.save(dataset / "set.000" / "coord.npy", coordinates)
    config = _config(tmp_path / "run", 1)
    config["model"]["bridging_method"] = "zbl"
    config["model"]["bridging_support"] = 1
    trainer = _run(config, dataset)
    for model in (trainer.model, trainer.ema_models["Default"]):
        clamp = model.atomic_model.models[0].descriptor.contact_clamp
        np.testing.assert_allclose(clamp.contacts[0, 1, 0], 0.8, rtol=1e-12)
        np.testing.assert_allclose(clamp.edges[0, 1], 0.8, rtol=1e-12)
        np.testing.assert_allclose(clamp.edges[1, 0], 0.8, rtol=1e-12)
    saved = BaseModel.deserialize(
        serialize_from_file(str(tmp_path / "run" / "model.jax"))["model"]
    )
    clamp = saved.atomic_model.models[0].descriptor.contact_clamp
    np.testing.assert_allclose(clamp.edges[0, 1], 0.8, rtol=1e-12)


def test_ema_checkpoint_uses_bias_adjusted_after_training(tmp_path: Path) -> None:
    """EMA changes parameters only and exports the live model's final buffers."""
    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 1)
    config["training"]["change_bias_after_training"] = True
    trainer = _run(config, dataset)
    live_bias = trainer.model.get_out_bias()
    np.testing.assert_array_equal(
        trainer.ema_models["Default"].get_out_bias(), live_bias
    )
    saved = BaseModel.deserialize(
        serialize_from_file(str(tmp_path / "run" / "model_ema.jax"))["model"]
    )
    np.testing.assert_array_equal(saved.get_out_bias(), live_bias)


def test_checkpoint_preserves_minimum_neighbor_distance(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 1)
    trainer = DPTrainer(deepcopy(config))
    trainer.set_min_nbor_dist(0.73)
    data = get_data({"systems": [str(dataset)], "batch_size": 1}, 4.0, ["O", "H"], None)
    trainer._setup_training({"Default": data}, {"Default": None})
    path = tmp_path / "minimum.jax"
    trainer._write_checkpoint(path, step=0)
    serialized = serialize_from_file(str(path))
    assert serialized["min_nbor_dist"] == 0.73
    assert serialized["model_def_script"]["_min_nbor_dist"] == 0.73
    for options in ({"restart": str(path)}, {"init_model": str(path)}):
        restored = DPTrainer(deepcopy(config), **options)
        assert restored.model.get_min_nbor_dist() == 0.73


@pytest.mark.parametrize("decoupled", [False, True])
def test_adam_task_activity_matches_pytorch(decoupled: bool) -> None:
    """Inactive heads retain moments and bias clocks across interleaved steps."""
    import torch

    initial = np.array([0.2, -0.3, 0.5])
    reference = [torch.nn.Parameter(torch.tensor(initial)) for _ in range(2)]
    optimizer_class = torch.optim.AdamW if decoupled else torch.optim.Adam
    reference_optimizer = optimizer_class(reference, lr=0.01, weight_decay=0.02)
    params = {str(index): jnp.asarray(initial) for index in range(2)}
    transform = build_adam(weight_decay=0.02, decoupled=decoupled)
    state = transform.init(params)
    update = jax.jit(transform.update)
    for step, selected in enumerate((0, 1, 0, 0, 1)):
        gradient = initial * (step + 1)
        reference_optimizer.zero_grad(set_to_none=True)
        reference[selected].grad = torch.tensor(gradient)
        reference_optimizer.step()
        active = {key: jnp.asarray(int(key) == selected) for key in params}
        grads = {
            key: jnp.asarray(gradient) if int(key) == selected else jnp.zeros(3)
            for key in params
        }
        updates, state = update(grads, state, params, learning_rate=0.01, active=active)
        params = optax.apply_updates(params, updates)
        for index in range(2):
            np.testing.assert_allclose(
                params[str(index)], reference[index].detach(), rtol=1e-12, atol=1e-12
            )


def test_shared_parameter_activity_uses_object_identity() -> None:
    """A shared parameter is active through every branch that owns it."""

    class Model(nnx.Module):
        def __init__(self, shared: nnx.Param) -> None:
            self.shared = shared
            self.head = nnx.Param(jnp.array(1.0))

    shared = nnx.Param(jnp.ones((2, 2)))
    models = {key: Model(shared) for key in ("a", "b")}
    masks = {
        key: mask.to_pure_dict() for key, mask in _task_parameter_masks(models).items()
    }
    assert bool(masks["a"]["a"]["shared"])
    assert bool(masks["b"]["a"]["shared"])
    assert bool(masks["a"]["a"]["head"])
    assert not bool(masks["a"]["b"]["head"])


@pytest.mark.parametrize("magnitude", [0.0, 1e-30, 1e30])
def test_gradient_clipping_matches_stable_pt_reduction(magnitude: float) -> None:
    import torch

    from deepmd.pt_expt.train.gradient import (
        clip_grad_norm_,
    )

    arrays = [
        np.array([magnitude, -magnitude], dtype=np.float32),
        np.array([0.2 * magnitude], dtype=np.float32),
    ]
    parameters = [
        torch.nn.Parameter(torch.zeros_like(torch.tensor(value))) for value in arrays
    ]
    for parameter, value in zip(parameters, arrays, strict=True):
        parameter.grad = torch.tensor(value)
    expected_norm = clip_grad_norm_(parameters, 0.5, stable=True)
    gradients = [jnp.asarray(value) for value in arrays]
    norm = jax.jit(_global_gradient_norm)(gradients)
    actual = jax.jit(lambda values: _clip_gradients(values, 0.5))(gradients)
    assert np.isfinite(norm)
    np.testing.assert_allclose(norm, expected_norm.numpy(), rtol=2e-7, atol=0.0)
    for value, parameter in zip(actual, parameters, strict=True):
        np.testing.assert_allclose(value, parameter.grad.numpy(), rtol=2e-7, atol=0.0)


def test_half_precision_gradient_norm_accumulates_without_overflow() -> None:
    import torch

    from deepmd.pt_expt.train.gradient import (
        clip_grad_norm_,
    )

    parameter = torch.nn.Parameter(torch.zeros(131072, dtype=torch.float16))
    parameter.grad = torch.ones_like(parameter)
    expected = clip_grad_norm_([parameter], 1.0, stable=True)
    gradients = [jnp.ones(131072, dtype=jnp.float16)]
    norm = jax.jit(_global_gradient_norm)(gradients)
    clipped = jax.jit(lambda values: _clip_gradients(values, 1.0))(gradients)
    np.testing.assert_array_equal(norm, expected.numpy())
    np.testing.assert_array_equal(clipped[0], parameter.grad.numpy())


def test_training_and_statistics_filter_without_filtering_validation(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "data")
    for path in (dataset / "set.000").glob("*.npy"):
        values = np.load(path)
        np.save(path, np.repeat(values, 2, axis=0))
    coordinates = np.load(dataset / "set.000" / "coord.npy")
    coordinates[0, 3:6] = [0.01, 0.0, 0.0]
    np.save(dataset / "set.000" / "coord.npy", coordinates)
    np.save(dataset / "set.000" / "energy.npy", [[1000.0], [0.3]])
    config = _config(tmp_path / "run", 1)
    config["training"]["training_data"] = {"min_pair_dist": 0.8}
    trainer = DPTrainer(config)
    data = get_data({"systems": [str(dataset)], "batch_size": 2}, 4.0, ["O", "H"], None)
    validation = get_data(
        {"systems": [str(dataset)], "batch_size": 2}, 4.0, ["O", "H"], None
    )
    trainer._setup_training({"Default": data}, {"Default": validation})
    assert trainer.pair_filter_enabled_by_task["Default"]
    for sample in trainer._sample_funcs["Default"]():
        np.testing.assert_allclose(sample["energy"], 0.3)
    result = trainer.train_step(TrainingTask(key="Default", training_data=data), 0)
    assert int(result.payload.labels["frame_mask"].sum()) == 1
    batch = validation.get_batch()
    assert "pair_margin" not in batch
    assert int(trainer._prepare_batch(batch, "Default").labels["frame_mask"].sum()) == 2


def test_nonfinite_gradient_cannot_overwrite_a_checkpoint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "data")
    trainer = DPTrainer(_config(tmp_path / "run", 1))
    data = get_data({"systems": [str(dataset)], "batch_size": 1}, 4.0, ["O", "H"], None)
    trainer._setup_training({"Default": data}, {"Default": None})
    checkpoint = tmp_path / "safe.jax"
    trainer._write_checkpoint(checkpoint, step=0)
    before = {
        str(path.relative_to(checkpoint)): path.stat().st_mtime_ns
        for path in checkpoint.rglob("*")
    }
    batch = trainer._prepare_batch(data.get_batch(), "Default")
    labels = dict(batch.labels)
    labels["force"] = jnp.full_like(labels["force"], jnp.nan)
    trainer.model.train()
    trainer._train_step_impls["Default"](
        trainer.models,
        trainer.optimizer,
        1e-3,
        batch._replace(labels=labels),
        trainer._active_parameters["Default"],
        trainer.ema_models,
    )
    assert bool(trainer.optimizer.nonfinite_gradient_seen[...])
    with pytest.raises(RuntimeError, match="non-finite gradient norm"):
        trainer._write_checkpoint(checkpoint, step=1)
    after = {
        str(path.relative_to(checkpoint)): path.stat().st_mtime_ns
        for path in checkpoint.rglob("*")
    }
    assert after == before


def test_full_validation_retry_does_not_replace_model_state(tmp_path: Path) -> None:
    """A failed inference dispatch leaves source arrays usable by smaller retries."""
    from deepmd.jax.train.validation import (
        JAXFullValidator,
    )

    model = get_model(model_config())
    validator = JAXFullValidator(
        validating_params={},
        validation_data=None,
        model=model,
        state_store={},
        num_steps=1,
        rank=0,
        restart_training=False,
        checkpoint_dir=tmp_path,
    )
    model.eval()
    identities = {
        path: id(variable.value) for path, variable in to_flat_state(nnx.state(model))
    }
    evaluate = validator._evaluate_batch
    calls = 0

    def fail_first_dispatch(
        current: BaseModel, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = evaluate(current, inputs)
        if calls == 1:
            raise RuntimeError("RESOURCE_EXHAUSTED: injected after inference dispatch")
        return result

    validator._evaluate_batch = fail_first_dispatch
    validator.auto_batch_size.current_batch_size = 12
    coordinates = np.tile([0.0, 0.0, 0.0, 1.1, 0.2, 0.3, 0.4, 1.3, 0.6], (4, 1))
    inputs = {
        "coord": coordinates,
        "atom_types": np.tile([0, 1, 0], (4, 1)),
        "box": None,
        "fparam": None,
        "aparam": None,
        "include_virial": False,
        "natoms": 3,
        "nframes": 4,
    }
    actual = validator._predict_outputs(**inputs)
    assert calls > 1
    after = {
        path: id(variable.value) for path, variable in to_flat_state(nnx.state(model))
    }
    assert after == identities
    validator._evaluate_batch = evaluate
    expected = validator._predict_outputs(**inputs)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


def test_structurally_unused_parameters_do_not_receive_weight_decay(
    tmp_path: Path,
) -> None:
    """A single-source attention aggregator retains its unused parameters."""
    torch = pytest.importorskip("torch")
    from deepmd.pt.model.model import get_model as get_pt_model

    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 2)
    config["model"]["descriptor"]["full_attn_res"] = "dependent"
    config["optimizer"] = {"type": "HybridMuon", "weight_decay": 0.1}
    reference = get_pt_model(deepcopy(config["model"])).cpu().train()
    prediction = reference(
        torch.tensor(
            [[[0.0, 0.0, 0.0], [1.1, 0.2, 0.3], [0.4, 1.3, 0.6]]], dtype=torch.float64
        ),
        torch.tensor([[0, 1, 0]]),
        charge_spin=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    )
    (prediction["energy"].sum() + prediction["force"].square().sum()).backward()
    reference_parameters = list(
        reference.get_descriptor().blocks[0].full_attn_res_so2.parameters()
    )
    assert reference_parameters and all(
        parameter.grad is None for parameter in reference_parameters
    )
    trainer = DPTrainer(config)
    data = get_data({"systems": [str(dataset)], "batch_size": 1}, 4.0, ["O", "H"], None)
    trainer._setup_training({"Default": data}, {"Default": None})
    inactive = trainer.model.get_descriptor().blocks[0].full_attn_res_so2
    before = [
        np.array(value, copy=True) for value in jax.tree.leaves(nnx.state(inactive))
    ]
    fitting_before = [
        np.array(value, copy=True)
        for value in jax.tree.leaves(
            nnx.state(trainer.model.get_fitting_net(), nnx.Param)
        )
    ]
    for step in range(2):
        trainer.train_step(TrainingTask(key="Default", training_data=data), step)
    for expected, actual in zip(
        before, jax.tree.leaves(nnx.state(inactive)), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert any(
        not np.array_equal(expected, actual)
        for expected, actual in zip(
            fitting_before,
            jax.tree.leaves(nnx.state(trainer.model.get_fitting_net(), nnx.Param)),
            strict=True,
        )
    )


def test_init_model_uses_target_runtime_and_fresh_random_stream(tmp_path: Path) -> None:
    """Loading weights honors target AMP, random rotations, and initialization seed."""
    source_config = model_config()
    source_config["descriptor"]["precision"] = "float32"
    source_config["fitting_net"]["precision"] = "float32"
    source = get_model(source_config)
    source_path = tmp_path / "source.jax"
    deserialize_to_file(
        str(source_path),
        {"model": source.serialize(), "model_def_script": source_config},
    )
    config = _config(tmp_path / "run", 1)
    config["model"] = deepcopy(source_config)
    config["model"]["descriptor"].update(use_amp=True, random_gamma=True, seed=20261024)
    expected = get_model(deepcopy(config["model"]))
    trainer = DPTrainer(config, init_model=str(source_path))
    descriptor = trainer.model.get_descriptor()
    assert descriptor.use_amp
    assert descriptor.random_gamma
    assert trainer.start_step == 0
    np.testing.assert_array_equal(
        descriptor.rotation_key.value, expected.get_descriptor().rotation_key.value
    )
    assert not np.array_equal(
        descriptor.rotation_key.value, source.get_descriptor().rotation_key.value
    )
    for actual, original in zip(
        jax.tree.leaves(nnx.state(trainer.model, nnx.Param)),
        jax.tree.leaves(nnx.state(source, nnx.Param)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, original)
    config["model"]["descriptor"]["channels"] = 8
    with pytest.raises(ValueError, match="expects shape"):
        DPTrainer(config, init_model=str(source_path))


@pytest.mark.parametrize("mode", ["init", "finetune"])
@pytest.mark.parametrize("target_rank", [None, 1])
def test_pretrained_lora_delta_survives_precision_and_rank_changes(
    mode: str, target_rank: int | None
) -> None:
    from deepmd.jax.descriptor.lora import (
        has_lora,
    )

    source_config = model_config()
    source_config["descriptor"]["precision"] = "float32"
    source_config["fitting_net"]["precision"] = "float32"
    source_config["lora"] = {"rank": 2, "alpha": 3.0}
    source = get_model(source_config)
    source_parameters = nnx.state(source, nnx.Param)
    for path, variable in to_flat_state(source_parameters):
        if any(str(part).startswith("B_") for part in path):
            variable.value = jnp.full_like(variable.value, 0.04)
    nnx.update(source, source_parameters)
    original = {
        path: np.array(variable.value, copy=True)
        for path, variable in to_flat_state(nnx.state(source))
    }
    expected = {
        path: np.asarray(variable.value)
        for path, variable in to_flat_state(
            nnx.state(merged_initialization_source(source))
        )
    }
    assert any(
        not np.array_equal(value, original[path]) for path, value in expected.items()
    )
    target_config = model_config()
    target_config["descriptor"]["seed"] = 20261024
    if target_rank is not None:
        target_config["lora"] = {"rank": target_rank, "alpha": float(target_rank)}
    target = get_model(target_config)
    target_key = np.asarray(target.get_descriptor().rotation_key.value).copy()
    copied = (
        DPTrainer._copy_initial_model_state(target, source)
        if mode == "init"
        else DPTrainer._copy_finetune_state(target, source, random_fitting=False)
    )
    actual = dict(to_flat_state(nnx.state(copied)))
    for path, value in expected.items():
        if path[-1] != "rotation_key":
            np.testing.assert_array_equal(
                actual[path].value, value.astype(actual[path].value.dtype)
            )
    np.testing.assert_array_equal(
        copied.get_descriptor().rotation_key.value, target_key
    )
    assert has_lora(copied) is (target_rank is not None)
    for path, variable in to_flat_state(nnx.state(source)):
        np.testing.assert_array_equal(variable.value, original[path])


def test_native_hybrid_checkpoint_round_trip_restarts_training(tmp_path: Path) -> None:
    """A native JAX training checkpoint retains optimizer and EMA through PT."""
    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 1)
    config["optimizer"] = {"type": "HybridMuon"}
    trained = _run(config, dataset)
    pt_path = tmp_path / "converted.pt"
    restored_path = tmp_path / "restored.jax"
    convert_backend(INPUT=str(tmp_path / "run" / "model.jax"), OUTPUT=str(pt_path))
    convert_backend(INPUT=str(pt_path), OUTPUT=str(restored_path))
    config["training"]["numb_steps"] = 2
    restored = DPTrainer(config, restart=str(restored_path))
    data = get_data({"systems": [str(dataset)], "batch_size": 1}, 4.0, ["O", "H"], None)
    restored._setup_training({"Default": data}, {"Default": None})
    _assert_states_equal(trained.models, restored.models)
    _assert_states_equal(trained.optimizer, restored.optimizer)
    _assert_states_equal(trained.ema_models, restored.ema_models)
    for run in (trained, restored):
        run.train_step(TrainingTask(key="Default", training_data=data), 1)
    _assert_states_equal(trained.models, restored.models)
    _assert_states_equal(trained.optimizer, restored.optimizer)
    _assert_states_equal(trained.ema_models, restored.ema_models)


@pytest.mark.parametrize("intensive", [False, True])
def test_dpa4_property_training(tmp_path: Path, intensive: bool) -> None:
    """Invariant property heads train through the shared graph trainer."""
    dataset = _dataset(tmp_path / "data")
    for path in (dataset / "set.000").glob("*.npy"):
        values = np.load(path)
        np.save(path, np.repeat(values, 2, axis=0))
    np.save(dataset / "set.000" / "property.npy", [[0.3, 0.6], [0.5, 0.9]])
    config = _config(tmp_path / "run", 2)
    config["model"]["data_stat_nbatch"] = 2
    config["model"]["fitting_net"] = {
        "type": "property",
        "property_name": "property",
        "task_dim": 2,
        "neuron": [4],
        "intensive": intensive,
        "precision": "float64",
        "seed": 20261023,
    }
    config["loss"] = {"type": "property", "loss_func": "mse", "metric": ["mae", "rmse"]}
    initial = get_model(deepcopy(config["model"]))
    trainer = _run(config, dataset)
    assert int(trainer.optimizer.step.value) == 2
    assert any(
        np.any(np.asarray(left) != np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(nnx.state(initial, nnx.Param)),
            jax.tree.leaves(nnx.state(trainer.model, nnx.Param)),
            strict=True,
        )
    )
    assert np.isfinite(np.loadtxt(tmp_path / "run" / "lcurve.out")).all()


@pytest.mark.parametrize("loss_func", ["smooth_mae", "mae", "mse", "rmse", "mape"])
@pytest.mark.parametrize("intensive", [False, True])
def test_property_loss_ignores_graph_capacity_padding(
    loss_func: str, intensive: bool
) -> None:
    """Flattened nodes and dummy frames preserve rectangular property losses."""
    loss = PropertyLoss(
        task_dim=2,
        var_name="property",
        loss_func=loss_func,
        metric=["mae", "rmse"],
        intensive=intensive,
    )
    prediction = jnp.array([[1.2, 1.5], [2.1, 2.2], [99.0, 99.0]])
    labels = {"property": jnp.array([[0.2, 0.4], [0.7, 0.6], [0.0, 0.0]])}
    expected, metrics = loss(
        1.0,
        2,
        {"property": prediction[:2], "mask": jnp.array([[1, 1], [1, 0]])},
        {"property": labels["property"][:2]},
    )

    def evaluate(value: jnp.ndarray) -> jnp.ndarray:
        return loss(
            1.0,
            4,
            {
                "property": value,
                "mask": jnp.array([1, 1, 1, 0]),
                "n_node": jnp.array([2, 1, 0]),
            },
            {**labels, "frame_mask": jnp.array([True, True, False])},
        )[0]

    actual, gradient = jax.jit(jax.value_and_grad(evaluate))(prediction)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[-1], 0.0)
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("loss_func", ["mae", "rmse"])
def test_property_exact_fit_has_zero_gradient(loss_func: str) -> None:
    """The zero-error subgradient neither changes parameters nor produces NaN."""
    loss = PropertyLoss(task_dim=1, var_name="property", loss_func=loss_func)
    labels = {"property": jnp.array([[0.5]])}

    def evaluate(value: jnp.ndarray) -> jnp.ndarray:
        return loss(1.0, 2, {"property": value}, labels)[0]

    value, gradient = jax.value_and_grad(evaluate)(labels["property"])
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(gradient, 0.0)


@pytest.mark.parametrize("explicit", [False, True])
def test_property_normalization_matches_pt_model_statistics(explicit: bool) -> None:
    from types import (
        SimpleNamespace,
    )

    import torch

    from deepmd.pt.loss.property import PropertyLoss as PTPropertyLoss
    from deepmd.pt.utils.env import (
        DEVICE,
    )

    prediction = np.array([[2.0, 5.0], [4.0, 8.0]])
    reference = np.array([[1.0, 4.0], [3.0, 5.0]])
    standard_deviation = np.array([2.0, 4.0])
    bias = np.array([1.0, -0.3])
    options = {"task_dim": 2, "var_name": "property", "loss_func": "mse"}
    if explicit:
        options.update(out_std=[1.0, 1.0], out_bias=[0.0, 0.0])
    pt_prediction = torch.tensor(prediction, device=DEVICE, requires_grad=True)

    class Model:
        atomic_model = SimpleNamespace(
            out_std=torch.tensor(standard_deviation, device=DEVICE).reshape(1, 1, 2),
            out_bias=torch.tensor(bias, device=DEVICE).reshape(1, 1, 2),
        )

        def __call__(self) -> dict[str, torch.Tensor]:
            return {"property": pt_prediction}

    _, pt_loss, _ = PTPropertyLoss(**options)(
        {},
        Model(),
        {"property": torch.tensor(reference, device=DEVICE)},
        natoms=2,
        learning_rate=1.0,
    )
    pt_loss.backward()
    loss = PropertyLoss(**options)

    def evaluate(value: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        return loss(
            1.0,
            2,
            {"property": value, "_out_std": std, "_out_bias": jnp.asarray(bias)},
            {"property": jnp.asarray(reference)},
        )[0]

    objective = jax.jit(jax.value_and_grad(evaluate))
    value, gradient = objective(
        jnp.asarray(prediction), jnp.asarray(standard_deviation)
    )
    np.testing.assert_allclose(
        value, pt_loss.detach().cpu().numpy(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        gradient, pt_prediction.grad.cpu().numpy(), rtol=1e-12, atol=1e-12
    )
    changed, _ = objective(jnp.asarray(prediction), jnp.asarray(2 * standard_deviation))
    np.testing.assert_allclose(
        changed, value if explicit else value / 4, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("bridging", [False, True])
def test_multitask_shared_descriptor_has_one_optimizer_trajectory(
    tmp_path: Path, bridging: bool
) -> None:
    """Alternating tasks update one backbone while leaving the inactive head fixed."""
    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 2)
    branch = config.pop("model")
    if bridging:
        branch["bridging_method"] = "zbl"
    config["model"] = {
        "shared_dict": {
            "type_map": branch["type_map"],
            "descriptor": branch["descriptor"],
        },
        "model_dict": {
            key: {
                **deepcopy(branch),
                "type_map": "type_map",
                "descriptor": "descriptor",
            }
            for key in ("a", "b")
        },
    }
    loss = config.pop("loss")
    config["loss_dict"] = {key: deepcopy(loss) for key in ("a", "b")}
    config["training"]["data_dict"] = {key: {} for key in ("a", "b")}
    trainer = DPTrainer(config)
    data = {
        key: get_data(
            {"systems": [str(dataset)], "batch_size": 1}, 4.0, ["O", "H"], None
        )
        for key in ("a", "b")
    }
    trainer._setup_training(data, dict.fromkeys(data))
    assert trainer.models["a"].get_descriptor() is trainer.models["b"].get_descriptor()
    before_b = nnx.clone(trainer.models["b"].get_fitting_net())
    trainer.train_step(TrainingTask(key="a", training_data=data["a"]), 0)
    _assert_states_equal(before_b, trainer.models["b"].get_fitting_net())
    before_a = nnx.clone(trainer.models["a"].get_fitting_net())
    trainer.train_step(TrainingTask(key="b", training_data=data["b"]), 1)
    _assert_states_equal(before_a, trainer.models["a"].get_fitting_net())
    assert int(trainer.optimizer.step.value) == 2
    counts = nnx.state(trainer.optimizer).to_pure_dict()["opt_state"]["count"]
    for path, value in jax.tree_util.tree_flatten_with_path(counts)[0]:
        expected = (
            2 if any(getattr(part, "key", None) == "descriptor" for part in path) else 1
        )
        assert int(value) == expected
    trainer._save_checkpoint(2)
    expanded_config = deepcopy(config)
    expanded_config["model"] = deepcopy(trainer.model_def_script)
    expanded_config["training"]["numb_steps"] = 3
    resumed = DPTrainer(expanded_config, restart=str(tmp_path / "run" / "model.jax"))
    resumed._setup_training(data, dict.fromkeys(data))
    assert resumed.models["a"].get_descriptor() is resumed.models["b"].get_descriptor()
    for run in (trainer, resumed):
        run.train_step(TrainingTask(key="a", training_data=data["a"]), 2)
    _assert_states_equal(trainer.models, resumed.models)
    _assert_states_equal(trainer.optimizer, resumed.optimizer)
    _assert_states_equal(trainer.ema_models, resumed.ema_models)


def test_shared_fitting_preserves_distinct_case_film_inputs(tmp_path: Path) -> None:
    """Shared fitting weights receive the sorted one-hot condition of each head."""
    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 2)
    branch = config.pop("model")
    branch["fitting_net"].update(dim_case_embd=2, case_film_embd=True)
    config["model"] = {
        "shared_dict": {
            "type_map": branch["type_map"],
            "descriptor": branch["descriptor"],
            "fitting_net": branch["fitting_net"],
        },
        "model_dict": {
            key: {
                **deepcopy(branch),
                "type_map": "type_map",
                "descriptor": "descriptor",
                "fitting_net": "fitting_net",
            }
            for key in ("b", "a")
        },
    }
    loss = config.pop("loss")
    config["loss_dict"] = {key: deepcopy(loss) for key in ("b", "a")}
    config["training"]["data_dict"] = {key: {} for key in ("b", "a")}
    trainer = DPTrainer(config)
    data = {
        key: get_data(
            {"systems": [str(dataset)], "batch_size": 1}, 4.0, ["O", "H"], None
        )
        for key in ("b", "a")
    }
    trainer._setup_training(data, dict.fromkeys(data))
    left = trainer.models["a"].get_fitting_net()
    right = trainer.models["b"].get_fitting_net()
    assert left.nets is right.nets
    np.testing.assert_array_equal(left.case_embd, [1.0, 0.0])
    np.testing.assert_array_equal(right.case_embd, [0.0, 1.0])
    for step, key in enumerate(("a", "b")):
        result = trainer.train_step(
            TrainingTask(key=key, training_data=data[key]), step
        )
        assert all(np.isfinite(value) for value in result.train_results.values())
    np.testing.assert_array_equal(left.case_embd, [1.0, 0.0])
    np.testing.assert_array_equal(right.case_embd, [0.0, 1.0])


def test_lora_best_checkpoint_merges_without_mutating_training_model(
    tmp_path: Path,
) -> None:
    """Best checkpoints contain ordinary weights and latest checkpoints retain adapters."""
    from deepmd.jax.descriptor.lora import (
        has_lora,
    )

    dataset = _dataset(tmp_path / "data")
    config = _config(tmp_path / "run", 1)
    source_config = deepcopy(config["model"])
    source = get_model(source_config)
    source_path = tmp_path / "source.jax"
    deserialize_to_file(
        str(source_path),
        {"model": source.serialize(), "model_def_script": source_config},
    )
    config["model"]["lora"] = {"rank": 2, "alpha": 2.0}
    config["optimizer"] = {"type": "HybridMuon"}
    config["validating"] = {
        "full_validation": True,
        "validation_freq": 1,
        "full_val_start": 0.0,
        "full_val_file": str(tmp_path / "run" / "val.log"),
        "save_best_dir": str(tmp_path / "best"),
    }
    trainer = _run(config, dataset, init_model=str(source_path))
    assert has_lora(trainer.model)
    best_path = next((tmp_path / "best").glob("best.ckpt-*.jax"))
    best_data = serialize_from_file(str(best_path))
    latest_data = serialize_from_file(str(tmp_path / "run" / "model.jax"))
    merged = BaseModel.deserialize(best_data["model"])
    assert not has_lora(merged)
    assert "lora" not in best_data["model_def_script"]
    assert has_lora(BaseModel.deserialize(latest_data["model"]))
    assert not (best_path / "training_state").exists()
    coord = jnp.array([[[0.0, 0.0, 0.0], [1.1, 0.2, 0.3], [0.4, 1.3, 0.6]]])
    atype = jnp.array([[0, 1, 0]])
    charge_spin = jnp.array([[0.0, 1.0]])
    expected = trainer.model(coord, atype, charge_spin=charge_spin)
    actual = merged(coord, atype, charge_spin=charge_spin)
    for key in ("energy", "force", "virial"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10)
    config["training"]["numb_steps"] = 2
    resumed = _run(config, dataset, restart=str(tmp_path / "run" / "model.jax"))
    continuous_config = deepcopy(config)
    continuous_config["training"]["save_ckpt"] = str(tmp_path / "continuous" / "model")
    continuous_config["training"]["save_dir"] = str(
        tmp_path / "continuous" / "checkpoints"
    )
    continuous_config["training"]["disp_file"] = str(
        tmp_path / "continuous" / "lcurve.out"
    )
    continuous_config.pop("validating")
    continuous = _run(continuous_config, dataset, init_model=str(source_path))
    _assert_states_equal(continuous.models, resumed.models)
    _assert_states_equal(continuous.optimizer, resumed.optimizer)
    _assert_states_equal(continuous.ema_models, resumed.ema_models)
