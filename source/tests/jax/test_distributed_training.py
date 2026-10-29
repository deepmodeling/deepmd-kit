# SPDX-License-Identifier: LGPL-3.0-or-later
"""Global-batch equivalence and optimizer-state placement over local devices."""

from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from typing import (
    Any,
)

import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.jax.env import (
    OptState,
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.jax.optimizer import (
    build_adam,
    build_hybrid_muon,
)
from deepmd.jax.train.distributed import (
    DistributedTraining,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
    PreparedBatch,
    TrainingTask,
    _batch_atom_capacity,
    _clip_gradients,
    _evaluate_inputs,
    _task_parameter_masks,
)
from deepmd.jax.utils.checkpoint_state import (
    pack_checkpoint_state,
    restore_checkpoint_state,
)
from deepmd.jax.utils.graph_batch import (
    prepare_graph_batch,
    prepare_graph_batches,
)

MULTI_DEVICE = jax.local_device_count() > 1


def _model_config() -> dict[str, Any]:
    return {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4",
            "sel": 8,
            "rcut": 3.0,
            "channels": 4,
            "n_radial": 4,
            "lmax": 1,
            "mmax": 1,
            "n_blocks": 1,
            "grid_branch": 0,
            "ffn_neurons": 4,
            "random_gamma": False,
            "precision": "float64",
            "seed": 81,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [4],
            "precision": "float64",
            "seed": 82,
        },
    }


def _raw_batch() -> dict[str, Any]:
    rng = np.random.default_rng(19)
    return {
        "atype": np.array([0, 1, 0, 1, 1, 0, 1, 0, 1]),
        "n_node": np.array([2, 3, 4]),
        "coord": rng.normal(size=(9, 3)),
        "energy": rng.normal(size=(3, 1)),
        "force": rng.normal(size=(9, 3)),
        "virial": rng.normal(size=(3, 9)),
        "find_energy": True,
        "find_force": True,
        "find_virial": True,
        "find_box": False,
    }


def _batch() -> tuple[dict[str, Any], dict[str, Any]]:
    return prepare_graph_batch(_raw_batch(), rcut=3.0, ntypes=2)


def _step(distributed: DistributedTraining, *, group_count: int = 1) -> Any:
    objective = EnergyLoss(
        starter_learning_rate=0.001,
        start_pref_e=1.0,
        limit_pref_e=1.0,
        start_pref_f=1.0,
        limit_pref_f=1.0,
        start_pref_v=0.2,
        limit_pref_v=0.2,
    )

    def loss(models: Any, batch: PreparedBatch, rate: Any) -> Any:
        batches = (batch,) if group_count == 1 else batch
        values = []
        for current in batches:
            prediction = _evaluate_inputs(models["task"], current.inputs)
            values.append(
                distributed.evaluate_loss(
                    objective,
                    rate,
                    _batch_atom_capacity(current.inputs),
                    prediction,
                    current.labels,
                )[0]
            )
        return sum(values) / group_count

    @distributed.jit_step
    def step(
        models: Any,
        optimizer: nnx.Optimizer,
        learning_rate: Any,
        batch: PreparedBatch,
        active: Any,
        ema_models: Any,
    ) -> Any:
        value, gradients = nnx.value_and_grad(loss)(models, batch, learning_rate)
        gradients = distributed.constrain_gradients(gradients)
        norm = optax.global_norm(gradients)
        clipped = _clip_gradients(gradients, 0.05)
        optimizer.update(models, clipped, learning_rate=learning_rate, active=active)
        shadow = jax.tree.map(
            lambda average, current: average + 0.2 * (current - average),
            nnx.state(ema_models, nnx.Param),
            nnx.state(models, nnx.Param),
        )
        nnx.update(ema_models, shadow)
        return {"loss": value, "norm": norm, "gradients": gradients}

    return step


def _assert_trees_close(left: Any, right: Any, *, atol: float = 2e-9) -> None:
    left_values, left_tree = jax.tree_util.tree_flatten_with_path(left)
    right_values, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for (path, first), second in zip(left_values, right_values, strict=True):
        np.testing.assert_allclose(
            first, second, atol=atol, rtol=2e-8, err_msg=str(path)
        )


@pytest.mark.skipif(
    not MULTI_DEVICE, reason="At least two local JAX devices are required"
)
@pytest.mark.parametrize("stage", [0, 1])
def test_global_graph_batch_matches_single_device(stage: int) -> None:
    distributed = DistributedTraining({"zero_stage": stage})
    single = DistributedTraining({}, devices=[jax.local_devices()[0]])
    reference = {"task": get_model(_model_config())}
    models = nnx.clone(reference)
    reference_ema, ema = nnx.clone(reference), nnx.clone(models)
    reference_optimizer = nnx.Optimizer(reference, build_hybrid_muon(), wrt=nnx.Param)
    optimizer = nnx.Optimizer(
        models,
        build_hybrid_muon(matrix_sharding=distributed.data_sharding()),
        wrt=nnx.Param,
    )
    active = _task_parameter_masks(models)["task"]
    reference_active = _task_parameter_masks(reference)["task"]
    labels, inputs = _batch()
    batch = PreparedBatch(*distributed.prepare_batch(labels, inputs))
    reference_batch = PreparedBatch(*single.prepare_batch(labels, inputs))
    distributed.place_state(models, optimizer, ema, active=active)
    single.place_state(
        reference, reference_optimizer, reference_ema, active=reference_active
    )
    distributed_step, single_step = _step(distributed), _step(single)
    for index in range(3):
        rate = 0.001 * 0.8**index
        actual = distributed_step(models, optimizer, rate, batch, active, ema)
        expected = single_step(
            reference,
            reference_optimizer,
            rate,
            reference_batch,
            reference_active,
            reference_ema,
        )
        _assert_trees_close(actual, expected)
        _assert_trees_close(nnx.state(models), nnx.state(reference), atol=2e-7)
        _assert_trees_close(nnx.state(ema), nnx.state(reference_ema), atol=2e-7)
    parameter_arrays = jax.tree_util.tree_leaves(nnx.state(models, nnx.Param))
    assert all(value.sharding.is_fully_replicated for value in parameter_arrays)
    optimizer_arrays = jax.tree_util.tree_leaves(nnx.state(optimizer))
    if stage == 1:
        assert any(not value.sharding.is_fully_replicated for value in optimizer_arrays)
        assert sum(
            value.addressable_shards[0].data.nbytes for value in optimizer_arrays
        ) < sum(value.nbytes for value in optimizer_arrays)
    else:
        assert all(value.sharding.is_fully_replicated for value in optimizer_arrays)
    assert np.asarray(batch.inputs["atype"] >= 0).sum() == 9
    assert np.asarray(batch.labels["frame_mask"]).sum() == 3


@pytest.mark.skipif(
    not MULTI_DEVICE, reason="At least two local JAX devices are required"
)
def test_unsupported_parameter_sharding_is_explicit() -> None:
    for stage in (2, 3):
        with pytest.raises(NotImplementedError, match="parameter sharding"):
            DistributedTraining({"zero_stage": stage})


def test_single_device_policy_and_batch() -> None:
    distributed = DistributedTraining(
        {"zero_stage": 1}, devices=[jax.local_devices()[0]]
    )
    assert distributed.policy.stage == 0
    labels, inputs = _batch()
    placed_labels, placed_inputs = distributed.prepare_batch(
        deepcopy(labels), deepcopy(inputs)
    )
    for key, value in inputs.items():
        np.testing.assert_array_equal(placed_inputs[key], value)
    for key, value in labels.items():
        np.testing.assert_array_equal(placed_labels[key], value)


def test_multidevice_version_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax, "__version__", "0.6.2")
    device = jax.local_devices()[0]
    with pytest.raises(NotImplementedError, match=r"requires JAX 0\.7\.2"):
        DistributedTraining({}, devices=[device, device])
    single = DistributedTraining({}, devices=[device])
    assert not single.is_distributed


@pytest.mark.skipif(
    not MULTI_DEVICE, reason="At least two local JAX devices are required"
)
def test_frame_filter_retries_whole_device_group() -> None:
    class Data:
        def __init__(self, margins: list[list[float]]) -> None:
            self.margins = margins
            self.index = 0

        def get_batch(self) -> dict[str, Any]:
            index = self.index
            self.index += 1
            margins = self.margins[index]
            return {
                "type": np.zeros((len(margins), 1), dtype=np.int64),
                "coord": np.zeros((len(margins), 1, 3)),
                "energy": np.arange(len(margins))[:, None] + index * 10,
                "pair_margin": np.asarray(margins),
                "find_pair_margin": True,
            }

        def get_batch_pass_length(self) -> int:
            return len(self.margins)

    distributed = DistributedTraining({}, devices=jax.local_devices()[:2])
    data = Data([[0.5], [2.0], [2.0, 0.5], [2.0]])
    first, second = distributed.get_batch_group(
        data, task_key="task", filter_enabled=True
    )
    assert data.index == 4
    np.testing.assert_array_equal(first["energy"], [[20]])
    np.testing.assert_array_equal(second["energy"], [[30]])
    invalid = Data([[0.5], [2.0], [0.5], [2.0]])
    with pytest.raises(RuntimeError, match="No globally valid training batch group"):
        distributed.get_batch_group(invalid, task_key="task", filter_enabled=True)
    assert invalid.index == 4


def test_validation_weights_chief_batches_by_physical_atoms() -> None:
    distributed = DistributedTraining({})

    class Data:
        def __init__(self) -> None:
            self.index = 0

        def get_batch(self) -> dict[str, Any]:
            round_index, group = divmod(self.index, distributed.device_count)
            self.index += 1
            count = (2, 6)[round_index]
            return {
                "type": np.array([[*([0] * count), -1]]),
                "coord": np.zeros((1, count + 1, 3)),
                "energy": np.array([[10.0 + round_index * 10 + group * 1000]]),
                "pair_margin": np.array([0.5]),
                "find_pair_margin": True,
            }

    data = Data()
    trainer = SimpleNamespace(
        distributed=distributed,
        _prepare_batch=lambda batches, task_key, **kwargs: batches,
        _evaluate_prepared_batch=lambda task_key, step, batches: {
            "mae_e": float(batches[0]["energy"][0, 0]),
        },
    )
    task = TrainingTask(
        key="Default", training_data=None, validation_data=data, valid_numb_batch=2
    )
    result = DPTrainer.evaluate_validation(trainer, task, 0, None)
    assert result == {"mae_e": 17.5}
    assert data.index == 2 * distributed.device_count


@pytest.mark.skipif(
    not MULTI_DEVICE, reason="At least two local JAX devices are required"
)
def test_device_groups_match_independent_rank_gradients() -> None:
    """Unequal local batches and label availability retain each rank's loss scale."""
    distributed = DistributedTraining(
        {"zero_stage": 1}, devices=jax.local_devices()[:2]
    )
    single = DistributedTraining({}, devices=[jax.local_devices()[0]])
    raw = _raw_batch()
    local_batches = []
    reference_batches = []
    for group, (frames, nodes) in enumerate(
        ((slice(0, 2), slice(0, 5)), (slice(2, 3), slice(5, 9)))
    ):
        local = {}
        for key, value in raw.items():
            if key in ("atype", "coord", "force"):
                local[key] = value[nodes]
            elif key in ("n_node", "energy", "virial"):
                local[key] = value[frames]
            else:
                local[key] = value
        local["find_force"] = group == 0
        local["find_virial"] = group == 1
        local_batches.append(local)
        local_labels, local_inputs = prepare_graph_batch(local, rcut=3.0, ntypes=2)
        reference_batches.append(
            PreparedBatch(*single.prepare_batch(local_labels, local_inputs))
        )
    labels, inputs = prepare_graph_batches(local_batches, rcut=3.0, ntypes=2)
    batch = PreparedBatch(*distributed.prepare_batch(labels, inputs))

    models = {"task": get_model(_model_config())}
    reference = nnx.clone(models)
    ema, reference_ema = nnx.clone(models), nnx.clone(reference)
    optimizer = nnx.Optimizer(
        models,
        build_hybrid_muon(matrix_sharding=distributed.data_sharding()),
        wrt=nnx.Param,
    )
    reference_optimizer = nnx.Optimizer(reference, build_hybrid_muon(), wrt=nnx.Param)
    active, reference_active = (
        _task_parameter_masks(models)["task"],
        _task_parameter_masks(reference)["task"],
    )
    distributed.place_state(models, optimizer, ema, active=active)
    single.place_state(
        reference, reference_optimizer, reference_ema, active=reference_active
    )
    step, reference_step = _step(distributed), _step(single, group_count=2)
    for index in range(2):
        rate = 0.001 * 0.8**index
        actual = step(models, optimizer, rate, batch, active, ema)
        expected = reference_step(
            reference,
            reference_optimizer,
            rate,
            tuple(reference_batches),
            reference_active,
            reference_ema,
        )
        _assert_trees_close(actual, expected)
        # The Newton-Schulz recurrence carries bfloat16 matrices, so a sharded
        # and a replicated orthogonalization may round an update element to
        # neighboring bfloat16 values; the bound admits one such step at
        # these learning rates.
        _assert_trees_close(nnx.state(models), nnx.state(reference), atol=2e-6)
        _assert_trees_close(nnx.state(ema), nnx.state(reference_ema), atol=2e-6)


@pytest.mark.skipif(
    not MULTI_DEVICE, reason="At least two local JAX devices are required"
)
def test_optimizer_model_alias_remains_replicated() -> None:
    """Legacy NNX optimizers may retain the model graph beside their moments."""

    class Model(nnx.Module):
        def __init__(self) -> None:
            self.weight = nnx.Param(jnp.ones((8, 8)))
            self.buffer = nnx.Variable(jnp.ones((8,)))

    model = Model()
    optimizer = nnx.Optimizer(model, build_adam(), wrt=nnx.Param)
    optimizer.model = model
    optimizer.nonfinite_gradient_seen = OptState(jnp.asarray(False))
    distributed = DistributedTraining({"zero_stage": 1})
    active = _task_parameter_masks({"task": model})
    distributed.place_state(model, optimizer, active=active)
    assert all(
        value.committed and value.sharding.is_fully_replicated
        for value in jax.tree.leaves(active)
    )
    assert all(
        value.sharding.is_fully_replicated
        for value in jax.tree.leaves(nnx.state(model))
    )
    assert any(
        not value.sharding.is_fully_replicated
        for value in jax.tree.leaves(nnx.state(optimizer, OptState))
    )

    @distributed.jit_step
    def step(
        model: Any, optimizer: Any, rate: Any, batch: Any, active: Any, ema: Any
    ) -> Any:
        del batch, active, ema
        gradients = nnx.grad(lambda current: jnp.sum(current.weight[...] ** 2))(model)
        optimizer.update(model, gradients, learning_rate=rate)
        return model.weight[...]

    step(model, optimizer, 0.01, None, None, None)
    assert optimizer.model is model
    assert all(
        value.sharding.is_fully_replicated
        for value in jax.tree.leaves(nnx.state(model))
    )


@pytest.mark.skipif(
    not MULTI_DEVICE, reason="At least two local JAX devices are required"
)
def test_shared_task_state_and_checkpoint(tmp_path: Path) -> None:
    """Inactive heads, shared momentum, EMA, and partitioned checkpoints agree."""

    class Model(nnx.Module):
        def __init__(self, shared: nnx.Param, seed: int) -> None:
            self.shared = shared
            self.head = nnx.Param(
                jnp.asarray(np.random.default_rng(seed).normal(size=(4,)))
            )

        def __call__(self, inputs: jax.Array) -> jax.Array:
            return (inputs @ self.shared[...]) @ self.head[...]

    def make_models() -> dict[str, Model]:
        shared = nnx.Param(jnp.asarray(np.random.default_rng(23).normal(size=(4, 4))))
        return {"a": Model(shared, 24), "b": Model(shared, 25)}

    distributed = DistributedTraining({"zero_stage": 1})
    single = DistributedTraining({}, devices=[jax.local_devices()[0]])
    models, reference = make_models(), make_models()
    ema, reference_ema = nnx.clone(models), nnx.clone(reference)
    optimizer = nnx.Optimizer(
        models, build_adam(weight_decay=0.03, decoupled=True), wrt=nnx.Param
    )
    reference_optimizer = nnx.Optimizer(
        reference, build_adam(weight_decay=0.03, decoupled=True), wrt=nnx.Param
    )
    active = _task_parameter_masks(models)
    reference_active = _task_parameter_masks(reference)
    distributed.place_state(models, optimizer, ema, active=active)
    single.place_state(
        reference, reference_optimizer, reference_ema, active=reference_active
    )
    inputs = np.random.default_rng(26).normal(size=(4 * distributed.device_count, 4))
    batch = jax.device_put(inputs, distributed.data_sharding())
    reference_batch = jax.device_put(inputs, single.replicated)

    def make_step(context: DistributedTraining, task: str) -> Any:
        @context.jit_step
        def step(
            models: Any,
            optimizer: nnx.Optimizer,
            rate: Any,
            batch: Any,
            mask: Any,
            ema: Any,
        ) -> Any:
            loss, gradients = nnx.value_and_grad(
                lambda tree: jnp.mean(tree[task](batch) ** 2)
            )(models)
            gradients = context.constrain_gradients(gradients)
            optimizer.update(
                models, _clip_gradients(gradients, 0.1), learning_rate=rate, active=mask
            )
            nnx.update(
                ema,
                jax.tree.map(
                    lambda average, value: average + 0.2 * (value - average),
                    nnx.state(ema, nnx.Param),
                    nnx.state(models, nnx.Param),
                ),
            )
            return loss

        return step

    steps = {task: make_step(distributed, task) for task in models}
    reference_steps = {task: make_step(single, task) for task in models}
    for index, task in enumerate(("a", "b", "a", "b")):
        rate = 0.001 * 0.8**index
        actual = steps[task](models, optimizer, rate, batch, active[task], ema)
        expected = reference_steps[task](
            reference,
            reference_optimizer,
            rate,
            reference_batch,
            reference_active[task],
            reference_ema,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        _assert_trees_close(nnx.state(models), nnx.state(reference), atol=1e-12)
        _assert_trees_close(
            nnx.state(optimizer), nnx.state(reference_optimizer), atol=1e-12
        )
        _assert_trees_close(nnx.state(ema), nnx.state(reference_ema), atol=1e-12)

    state = nnx.state((models, optimizer, ema))
    packed = pack_checkpoint_state(state.to_pure_dict())
    path = tmp_path / "distributed-state"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(path, packed)
        checkpointer.wait_until_finished()
        restored = checkpointer.restore(path, target=packed)
    restored_state = nnx.state((models, optimizer, ema))
    restored_state.replace_by_pure_dict(
        restore_checkpoint_state(restored, state.to_pure_dict())
    )
    resumed_models, resumed_optimizer, resumed_ema = nnx.clone((models, optimizer, ema))
    nnx.update((resumed_models, resumed_optimizer, resumed_ema), restored_state)
    assert resumed_models["a"].shared is resumed_models["b"].shared
    steps["a"](models, optimizer, 0.0003, batch, active["a"], ema)
    steps["a"](
        resumed_models, resumed_optimizer, 0.0003, batch, active["a"], resumed_ema
    )
    _assert_trees_close(
        nnx.state((models, optimizer, ema)),
        nnx.state((resumed_models, resumed_optimizer, resumed_ema)),
        atol=0.0,
    )


@pytest.mark.skipif(
    not MULTI_DEVICE, reason="At least two local JAX devices are required"
)
def test_native_spin_zbl_sharded_restart(tmp_path: Path) -> None:
    """A magnetic-force run restores sharded optimizer, augmentation, and EMA state."""
    from .test_dpa4_training import (
        _assert_states_equal,
        _config,
        _dataset,
        _run,
    )

    dataset = _dataset(tmp_path / "data")
    continuous_config = _config(tmp_path / "continuous", 3, spin=True)
    continuous_config["training"]["zero_stage"] = 1
    continuous_config["model"]["descriptor"]["random_gamma"] = True
    continuous = _run(continuous_config, dataset)
    partial_config = _config(tmp_path / "resumed", 2, spin=True)
    partial_config["training"]["zero_stage"] = 1
    partial_config["model"]["descriptor"]["random_gamma"] = True
    _run(partial_config, dataset)
    partial_config["training"]["numb_steps"] = 3
    resumed = _run(
        partial_config, dataset, restart=str(tmp_path / "resumed" / "model.jax")
    )
    assert resumed.start_step == 2
    _assert_states_equal(continuous.models, resumed.models)
    _assert_states_equal(continuous.optimizer, resumed.optimizer)
    _assert_states_equal(continuous.ema_models, resumed.ema_models)
    assert any(
        not value.sharding.is_fully_replicated
        for value in jax.tree_util.tree_leaves(nnx.state(resumed.optimizer))
    )
