# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility of shared NNX state and container conversions."""

from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

import numpy as np
import pytest
from packaging.version import (
    Version,
)

from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.utils.network import NativeLayer as NativeLayerDP
from deepmd.jax import (
    common,
)
from deepmd.jax.env import (
    OptState,
    flax_version,
    jax,
    jnp,
    nnx,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
    NativeLayer,
)


def test_pure_tree_supports_legacy_variable_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live variables and legacy snapshots unwrap without losing tree structure."""

    @dataclass
    class VariableState:
        value: Any

    monkeypatch.delattr(nnx, "as_pure", raising=False)
    monkeypatch.setattr(nnx, "VariableState", VariableState, raising=False)
    value = jnp.asarray([1.0, 2.0])
    source = {"live": nnx.Param(value), "saved": VariableState((nnx.Param(value), 3))}
    result = common.as_pure(source)
    np.testing.assert_array_equal(result["live"], value)
    np.testing.assert_array_equal(result["saved"][0], value)
    assert result["saved"][1] == 3
    assert isinstance(source["saved"], VariableState)


def test_legacy_lists_keep_their_parameter_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older NNX receives ordinary lists with the original parameter identities."""
    monkeypatch.setattr(common, "flax_version", "0.10.7")
    values = [nnx.Param(jnp.ones(2)), None]
    assert common._wrap_list(values) is values


def test_flat_state_round_trip_preserves_paths_types_and_metadata() -> None:
    """State conversion retains integer paths and the parameter/buffer distinction."""
    state = nnx.state(
        {
            "blocks": {0: {"weight": nnx.Param(jnp.asarray([1.0, 2.0]), tag="dense")}},
            "indices": common.ArrayAPIVariable(jnp.asarray([0, 1])),
        }
    )
    restored = common.from_flat_state(common.to_flat_state(state))
    flat = dict(common.to_flat_state(restored))
    assert set(flat) == {("blocks", 0, "weight"), ("indices",)}
    assert flat[("blocks", 0, "weight")].tag == "dense"
    assert [path for path, _ in common.to_flat_state(restored.filter(nnx.Param))] == [
        ("blocks", 0, "weight"),
    ]
    for path, variable in common.to_flat_state(state):
        np.testing.assert_array_equal(
            common.as_pure(flat[path]), common.as_pure(variable)
        )


@pytest.mark.parametrize("trainable", [False, True])
def test_native_layer_jit_preserves_variable_policy(trainable: bool) -> None:
    """Graph reconstruction preserves typed weights without rerunning conversion."""
    layer = NativeLayer(2, 3, precision="float64", trainable=trainable, seed=42)
    reference = NativeLayerDP.deserialize(layer.serialize())
    inputs = jnp.asarray([[0.2, -0.4], [0.3, 0.5]])

    @nnx.jit
    def evaluate(model: NativeLayer, values: jax.Array) -> jax.Array:
        return model(values)

    np.testing.assert_allclose(
        evaluate(layer, inputs), reference(np.asarray(inputs)), atol=1e-14
    )
    assert isinstance(layer.w, nnx.Param) is trainable
    assert layer.trainable is trainable


def test_compiled_array_conversion_preserves_parameter_gradients() -> None:
    """Wrapped indices enter JIT as arrays while values retain their derivatives."""

    class Model(nnx.Module):
        def __init__(self) -> None:
            self.weight = ArrayAPIParam(jnp.asarray([1.0, 2.0, 3.0]))
            self.indices = common.ArrayAPIVariable(jnp.asarray([0, 2, 2]))

    def loss(model: Model) -> jax.Array:
        values = xp_asarray_nodetach(jnp, model.weight)
        indices = xp_asarray_nodetach(jnp, model.indices)
        return jnp.sum(jnp.take(values, indices) ** 2)

    model = Model()
    value, gradients = nnx.jit(nnx.value_and_grad(loss))(model)
    np.testing.assert_allclose(value, 19.0)
    np.testing.assert_array_equal(common.as_pure(gradients)["weight"], [2.0, 0.0, 12.0])


def test_compiled_variable_replacement_preserves_state() -> None:
    """Whole-value replacement updates optimizer state on immutable JAX arrays."""

    class Counter(nnx.Module):
        def __init__(self) -> None:
            self.step = OptState(jnp.asarray(0))

    @nnx.jit
    def advance(counter: Counter) -> jax.Array:
        common.set_variable_value(counter.step, counter.step[...] + 1)
        return counter.step[...]

    counter = Counter()
    assert int(advance(counter)) == 1
    assert int(advance(counter)) == 2
    assert isinstance(counter.step, OptState)


@pytest.mark.skipif(
    Version(flax_version) >= Version("0.11.0"),
    reason="The model-owning optimizer API is specific to Flax before 0.11.",
)
def test_legacy_optimizer_updates_shared_models_and_preserves_inactive_heads() -> None:
    """Dictionary gradients and activity masks retain the optimizer state tree."""
    from deepmd.jax.optimizer import (
        build_hybrid_muon,
    )
    from deepmd.jax.train.trainer import (
        _legacy_optimizer_update,
        _task_parameter_masks,
    )

    class Model(nnx.Module):
        def __init__(self, weight: nnx.Param) -> None:
            self.weight = weight
            self.head = nnx.Param(jnp.asarray([0.4, -0.3]))

    shared = nnx.Param(jnp.asarray([[0.2, -0.1], [0.4, 0.3]]))
    models = {name: Model(shared) for name in ("first", "second")}
    optimizer = nnx.Optimizer(
        models,
        build_hybrid_muon(weight_decay=0.1),
        wrt=nnx.Param,
    )
    weight_before = np.asarray(shared.value).copy()
    head_before = np.asarray(models["second"].head.value).copy()
    gradients = nnx.grad(
        lambda values: (
            jnp.sum(values["first"].weight.value ** 2)
            + jnp.sum(values["first"].head.value ** 2)
        )
    )(models)
    _legacy_optimizer_update(
        optimizer,
        gradients,
        1e-3,
        active=_task_parameter_masks(models)["first"],
    )
    assert int(optimizer.step.value) == 1
    assert models["first"].weight is models["second"].weight
    assert not np.array_equal(shared.value, weight_before)
    assert not np.array_equal(models["first"].head.value, head_before)
    np.testing.assert_array_equal(models["second"].head.value, head_before)
