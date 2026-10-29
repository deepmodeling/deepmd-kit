# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unused parameters remain distinct from participating zero gradients."""

from typing import (
    Any,
)

import numpy as np
import pytest
from packaging.version import (
    Version,
)

from deepmd.jax.common import (
    as_pure,
    to_flat_state,
)
from deepmd.jax.env import (
    flax_version,
    jax,
    jnp,
    nnx,
)
from deepmd.jax.optimizer import (
    build_adam,
)
from deepmd.jax.train.parameter_activity import (
    parameter_activity,
)
from deepmd.jax.train.trainer import (
    _legacy_optimizer_update,
)


class _Model(nnx.Module):
    def __init__(self) -> None:
        self.active = nnx.Param(jnp.asarray(2.0))
        self.unused = nnx.Param(jnp.asarray(3.0))
        self.detached = nnx.Param(jnp.asarray(4.0))
        self.zero = nnx.Param(jnp.asarray(0.0))


@pytest.mark.parametrize("nested", [False, True])
def test_structural_activity_inside_jit(nested: bool) -> None:
    model = _Model()

    def objective(model: _Model, target: Any) -> Any:
        return (
            (model.active[...] - target) ** 2
            + model.zero[...] ** 2
            + jax.lax.stop_gradient(model.detached[...])
        )

    if nested:
        objective = nnx.jit(objective)

    @nnx.jit
    def activity(model: _Model, target: Any) -> Any:
        return parameter_activity(lambda current: objective(current, target), model)

    active = activity(model, jnp.asarray(2.0)).to_pure_dict()
    assert {key: bool(value) for key, value in active.items()} == {
        "active": True,
        "unused": False,
        "detached": False,
        "zero": True,
    }


def test_unused_parameters_skip_decay_and_moment_clocks() -> None:
    model = _Model()
    optimizer = nnx.Optimizer(
        model,
        build_adam(weight_decay=0.1, decoupled=True),
        wrt=nnx.Param,
    )

    @nnx.jit
    def step(model: _Model, optimizer: nnx.Optimizer) -> None:
        def objective(current: _Model) -> Any:
            return (current.active[...] - 2.0) ** 2 + current.zero[...] ** 2

        active = parameter_activity(objective, model)
        gradients = nnx.grad(objective)(model)
        if Version(flax_version) >= Version("0.11.0"):
            optimizer.update(model, gradients, learning_rate=0.01, active=active)
        else:
            _legacy_optimizer_update(optimizer, gradients, 0.01, active=active)

    initial = nnx.state(model).to_pure_dict()
    step(model, optimizer)
    np.testing.assert_allclose(model.active[...], 1.998)
    np.testing.assert_array_equal(model.unused[...], initial["unused"])
    np.testing.assert_array_equal(model.detached[...], initial["detached"])
    counts = nnx.state(optimizer).to_pure_dict()["opt_state"]["count"]
    assert {key: int(value) for key, value in counts.items()} == {
        "active": 1,
        "unused": 0,
        "detached": 0,
        "zero": 1,
    }


def test_activity_tracks_static_loss_specializations() -> None:
    model = _Model()

    @nnx.jit(static_argnames=("use_other",))
    def activity(model: _Model, *, use_other: bool) -> Any:
        def objective(current: _Model) -> Any:
            return current.unused[...] ** 2 if use_other else current.active[...] ** 2

        return parameter_activity(objective, model)

    first = activity(model, use_other=False).to_pure_dict()
    second = activity(model, use_other=True).to_pure_dict()
    assert bool(first["active"]) and not bool(first["unused"])
    assert bool(second["unused"]) and not bool(second["active"])


def test_non_scalar_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="scalar training objective"):
        parameter_activity(lambda model: jnp.stack([model.active[...]]), _Model())


@pytest.mark.parametrize("mode", ["full", "block", "zero"])
def test_dpa4_activity_matches_pytorch_gradient_presence(mode: str) -> None:
    import torch

    from deepmd.jax.model.base_model import (
        BaseModel,
    )

    from .test_dpa4_model import (
        _reference,
        model_config,
    )
    from .test_dpa4_model_audit import (
        _graph_inputs,
        _inputs,
    )

    config = model_config()
    if mode == "zero":
        config["descriptor"]["n_blocks"] = 0
    else:
        config["descriptor"][f"{mode}_attn_res"] = "dependent"
    reference = _reference(config).train()
    model = BaseModel.deserialize(reference.serialize())
    model.train()
    coord, atype, _ = _inputs()
    expected = reference(torch.tensor(coord), torch.tensor(atype))
    expected_loss = (
        (expected["energy"] - 0.2).square().sum()
        + 0.5 * (expected["force"] - 0.1).square().sum()
        + 0.1 * (expected["virial"] + 0.05).square().sum()
    )
    expected_loss.backward()
    inputs = _graph_inputs(model, coord, atype)

    def objective(current: Any) -> Any:
        output = current.call_common_lower_graph(**inputs)
        return (
            jnp.sum((output["energy_redu"] - 0.2) ** 2)
            + 0.5 * jnp.sum((output["energy_derv_r"] - 0.1) ** 2)
            + 0.1 * jnp.sum((output["energy_derv_c_redu"] + 0.05) ** 2)
        )

    active = parameter_activity(objective, model)
    with torch.no_grad():
        for parameter in reference.parameters():
            if parameter.requires_grad:
                parameter.fill_(float(parameter.grad is not None))
    marker = BaseModel.deserialize(reference.serialize())
    expected_state = nnx.state(marker, nnx.Param)
    assert active.keys() == expected_state.keys()
    for (path, variable), (expected_path, expected_value) in zip(
        to_flat_state(active),
        to_flat_state(expected_state),
        strict=True,
    ):
        assert path == expected_path
        assert np.all(np.asarray(as_pure(expected_value)) == bool(as_pure(variable))), (
            path
        )
    if mode != "zero":
        assert not all(bool(value) for value in jax.tree.leaves(active))
