# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 invariant-property models reuse the common JAX model and fitting APIs."""

from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import pytest
import torch

from deepmd.entrypoints.convert_backend import (
    convert_backend,
)
from deepmd.infer.deep_property import (
    DeepProperty,
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
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
)

from .test_dpa4_model import (
    _reference,
    model_config,
)
from .test_dpa4_model_audit import (
    _graph_inputs,
    _inputs,
    _variables,
)


def _property_config(intensive: bool) -> dict[str, Any]:
    config = model_config()
    config["fitting_net"] = {
        "type": "property",
        "task_dim": 2,
        "property_name": "observables",
        "neuron": [0],
        "intensive": intensive,
        "precision": "float64",
        "seed": 17,
    }
    config["atom_exclude_types"] = [1]
    return config


@pytest.mark.parametrize("intensive", [False, True])
def test_property_reduction_gradients_and_padding(intensive: bool) -> None:
    config = _property_config(intensive)
    reference = _reference(config)
    model = BaseModel.deserialize(reference.serialize())
    assert (
        get_model(config).get_fitting_net().neuron == reference.get_fitting_net().neuron
    )
    coord, atype, _ = _inputs()
    expected = reference(torch.tensor(coord), torch.tensor(atype))
    expected["observables"].square().sum().backward()
    inputs = _graph_inputs(model, coord, atype)
    inputs["n_node"] = jnp.pad(inputs["n_node"], (0, 1))

    def loss(model: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        output = model.call_common_lower_graph(**inputs)
        return jnp.sum(output["observables_redu"] ** 2), output

    (value, output), gradients = nnx.jit(nnx.value_and_grad(loss, has_aux=True))(model)
    np.testing.assert_allclose(
        output["observables_redu"][:1],
        expected["observables"].detach().numpy(),
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_array_equal(output["observables_redu"][1:], 0.0)
    np.testing.assert_allclose(
        output["observables"],
        expected["atom_observables"].detach().numpy().reshape(3, 2),
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        value,
        expected["observables"].detach().square().sum().numpy(),
        rtol=2e-9,
        atol=2e-10,
    )
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    assert any(np.linalg.norm(leaf) > 1e-8 for leaf in jax.tree.leaves(gradients))
    with torch.no_grad():
        for parameter in reference.parameters():
            if parameter.requires_grad:
                parameter.copy_(
                    parameter.grad
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
    graph, _, constants = nnx.split(model, nnx.Param, ...)
    gradient_model = nnx.merge(graph, gradients, constants)
    for method in ("get_descriptor", "get_fitting_net"):
        expected_variables = dict(_variables(getattr(reference, method)().serialize()))
        actual_variables = dict(
            _variables(getattr(gradient_model, method)().serialize())
        )
        assert expected_variables.keys() == actual_variables.keys()
        for name, expected_value in expected_variables.items():
            np.testing.assert_allclose(
                actual_variables[name],
                expected_value,
                rtol=3e-8,
                atol=3e-10,
                err_msg=name,
            )


@pytest.mark.parametrize("suffix", [".jax", ".hlo"])
@pytest.mark.parametrize("intensive", [False, True])
def test_property_checkpoint_inference(
    tmp_path: Path, suffix: str, intensive: bool
) -> None:
    config = _property_config(intensive)
    reference = _reference(config)
    path = tmp_path / f"property{suffix}"
    deserialize_to_file(
        str(path), {"model": reference.serialize(), "model_def_script": config}
    )
    coord, atype, _ = _inputs()
    expected = reference(torch.tensor(coord), torch.tensor(atype))
    actual = DeepProperty(str(path), auto_batch_size=False).eval(
        coord, None, atype, atomic=True, mixed_type=True
    )
    np.testing.assert_allclose(
        actual[0], expected["observables"].detach().numpy(), rtol=2e-9, atol=2e-10
    )
    np.testing.assert_allclose(
        actual[1], expected["atom_observables"].detach().numpy(), rtol=2e-9, atol=2e-10
    )


def test_property_checkpoint_converts_back_to_pt(tmp_path: Path) -> None:
    """The public backend converter preserves the SeZM property wrapper."""
    config = _property_config(True)
    reference = _reference(config)
    source = tmp_path / "source.jax"
    target = tmp_path / "restored.pt"
    deserialize_to_file(
        str(source), {"model": reference.serialize(), "model_def_script": config}
    )
    convert_backend(INPUT=str(source), OUTPUT=str(target))
    coord, atype, _ = _inputs()
    expected = reference(torch.tensor(coord), torch.tensor(atype))
    actual = DeepProperty(str(target), auto_batch_size=False).eval(
        coord, None, atype, atomic=True, mixed_type=True
    )
    np.testing.assert_allclose(
        actual[0], expected["observables"].detach().numpy(), rtol=2e-9, atol=2e-10
    )
    np.testing.assert_allclose(
        actual[1], expected["atom_observables"].detach().numpy(), rtol=2e-9, atol=2e-10
    )
