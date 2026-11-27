# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independent complete-model derivatives and physical properties of JAX DPA4."""

from collections.abc import (
    Iterator,
)
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
import pytest
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.utils.graph_batch import (
    prepare_graph_batch,
)
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
)

from .test_dpa4_model import (
    EXPORTABLE_OPTIONS,
    _reference,
    model_config,
)

OPTIONS = [
    pytest.param(
        {
            "l_schedule": [2, 1],
            "m_schedule": [1, 1],
            "full_attn_res": "dependent",
            "readout_layers": 2,
            "basis_type": "gaussian",
        },
        False,
        id="pyramid_full_attention",
    ),
    pytest.param(
        {
            "l_schedule": [2, 1],
            "m_schedule": [1, 1],
            "edge_cartesian": True,
            "node_cartesian": "parity:2",
            "block_attn_res": "independent",
            "so2_layers": 2,
        },
        False,
        id="cartesian_block_attention",
    ),
    pytest.param(
        {
            "lmax": 2,
            "mmax": 1,
            "extra_node_l": 1,
            "ffn_so3_grid": True,
            "message_node_so3": True,
            "so3_readout": "glu",
            "readout_layers": 2,
            "layer_scale": True,
        },
        False,
        id="node_so3_readout",
    ),
    pytest.param({"basis_type": "bessel/fix"}, True, id="spin_zbl_vacuum_exclusions"),
    pytest.param(
        {"n_blocks": 0, "lmax": 0, "mmax": 0, "kmax": 0}, False, id="zero_blocks"
    ),
]


def _variables(data: Any, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
    if isinstance(data, np.ndarray):
        yield prefix, data
    elif isinstance(data, dict):
        for key, value in data.items():
            yield from _variables(value, f"{prefix}.{key}")
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from _variables(value, f"{prefix}.{index}")


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord = np.array([[[0.2, 0.1, -0.1], [1.1, 0.4, 0.3], [0.5, 1.4, 0.8]]])
    atype = np.array([[0, 1, 0]], dtype=np.int32)
    spin = np.array([[[0.2, -0.3, 0.4], [0.0, 0.0, 0.0], [-0.3, 0.1, 0.2]]])
    return coord, atype, spin


def _graph_inputs(
    model: Any, coord: np.ndarray, atype: np.ndarray, spin: np.ndarray | None = None
) -> dict[str, jax.Array]:
    _, inputs = prepare_graph_batch(
        {"coord": coord, "type": atype, "spin": spin, "find_box": False},
        rcut=model.get_rcut(),
        ntypes=model.get_ntypes(),
        pair_exclude_types=model.atomic_model.pair_exclude_types,
    )
    return jax.tree.map(jnp.asarray, inputs)


@pytest.mark.parametrize("options,physics", OPTIONS)
def test_complete_force_loss_parameter_gradients(
    options: dict[str, Any], physics: bool
) -> None:
    """Every trainable model leaf matches the PT energy/force/virial loss gradient."""
    config = model_config(spin=physics, bridging=physics)
    config["descriptor"].update(options)
    if physics:
        config["fitting_net"]["vacuum_ref"] = True
        config["pair_exclude_types"] = [[0, 1]]
        config["atom_exclude_types"] = [1]
    reference = _reference(config).train()
    model = BaseModel.deserialize(reference.serialize())
    model.train()
    coord, atype, spin = _inputs()
    expected = reference(
        torch.tensor(coord),
        torch.tensor(atype),
        **({"spin": torch.tensor(spin)} if physics else {}),
    )
    expected_loss = (
        (expected["energy"] - 0.2).square().sum()
        + 0.5 * (expected["force"] - 0.1).square().sum()
        + 0.1 * (expected["virial"] + 0.05).square().sum()
    )
    if physics:
        expected_loss += 0.3 * (expected["force_mag"] - 0.02).square().sum()
    expected_loss.backward()
    inputs = _graph_inputs(model, coord, atype, spin if physics else None)
    graph, parameters, constants = nnx.split(model, nnx.Param, ...)
    nframes, nnodes = atype.shape[0], atype.size

    def loss(parameters: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        current = nnx.merge(graph, parameters, constants)
        output = current.call_common_lower_graph(**inputs)
        value = (
            jnp.sum((output["energy_redu"][:nframes] - 0.2) ** 2)
            + 0.5 * jnp.sum((output["energy_derv_r"][:nnodes] - 0.1) ** 2)
            + 0.1 * jnp.sum((output["energy_derv_c_redu"][:nframes] + 0.05) ** 2)
        )
        if physics:
            value += 0.3 * jnp.sum((output["energy_derv_r_mag"][:nnodes] - 0.02) ** 2)
        return value, output

    (actual_loss, output), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(
        parameters
    )
    np.testing.assert_allclose(
        actual_loss, expected_loss.detach().numpy(), rtol=2e-9, atol=2e-9
    )
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    for source, target in (
        ("energy", "energy_redu"),
        ("force", "energy_derv_r"),
        ("virial", "energy_derv_c_redu"),
    ):
        count = nframes if target.endswith("_redu") else nnodes
        values = np.asarray(output[target])
        np.testing.assert_array_equal(values[count:], 0.0)
        np.testing.assert_allclose(
            values[:count].reshape(expected[source].shape),
            expected[source].detach().numpy(),
            rtol=2e-9,
            atol=2e-10,
        )
    assert len(jax.tree.leaves(parameters)) == sum(
        parameter.requires_grad for parameter in reference.parameters()
    )
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(
                    parameter.grad
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
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
                rtol=3e-7,
                atol=3e-9,
                err_msg=f"{method}:{name}",
            )


@pytest.mark.parametrize("bridging", [False, True])
def test_exportable_options_match_pt_expt(bridging: bool) -> None:
    """Energies, derivatives and loss gradients equal those of PyTorch Exportable."""
    from deepmd.pt_expt.model import get_model as get_pt_expt_model

    config = model_config(bridging=bridging)
    config["descriptor"].update(EXPORTABLE_OPTIONS)
    reference = get_pt_expt_model(deepcopy(config)).cpu().train()
    generator = torch.Generator().manual_seed(20261107)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape, dtype=parameter.dtype, generator=generator
                )
            )
        # A measured envelope mass of every block cutoff sets its floor.
        for name, buffer in reference.named_buffers():
            if name.endswith("envelope_mass"):
                buffer.copy_(torch.tensor([3.0, 5.5], dtype=buffer.dtype))
    model = BaseModel.deserialize(reference.serialize())
    model.train()
    coord = np.concatenate([_inputs()[0], [[[2.6, 0.3, 0.2]]]], axis=1)
    atype = np.array([[0, 1, 0, 1]], dtype=np.int32)
    expected = reference(torch.tensor(coord), torch.tensor(atype))
    expected_loss = (
        (expected["energy"] - 0.2).square().sum()
        + 0.5 * (expected["force"] - 0.1).square().sum()
        + 0.1 * (expected["virial"] + 0.05).square().sum()
    )
    expected_loss.backward()
    inputs = _graph_inputs(model, coord, atype)
    graph, parameters, constants = nnx.split(model, nnx.Param, ...)
    nframes, nnodes = atype.shape[0], atype.size

    def loss(parameters: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        output = nnx.merge(graph, parameters, constants).call_common_lower_graph(
            **inputs
        )
        value = (
            jnp.sum((output["energy_redu"][:nframes] - 0.2) ** 2)
            + 0.5 * jnp.sum((output["energy_derv_r"][:nnodes] - 0.1) ** 2)
            + 0.1 * jnp.sum((output["energy_derv_c_redu"][:nframes] + 0.05) ** 2)
        )
        return value, output

    (actual_loss, output), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(
        parameters
    )
    np.testing.assert_allclose(
        actual_loss, expected_loss.detach().numpy(), rtol=2e-9, atol=2e-9
    )
    for source, target in (
        ("energy", "energy_redu"),
        ("force", "energy_derv_r"),
        ("virial", "energy_derv_c_redu"),
    ):
        count = nframes if target.endswith("_redu") else nnodes
        np.testing.assert_allclose(
            np.asarray(output[target])[:count].reshape(expected[source].shape),
            expected[source].detach().numpy(),
            rtol=2e-9,
            atol=2e-10,
        )
    assert len(jax.tree.leaves(parameters)) == sum(
        parameter.requires_grad for parameter in reference.parameters()
    )
    with torch.no_grad():
        for parameter in reference.parameters():
            if parameter.requires_grad:
                parameter.copy_(
                    parameter.grad
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
    gradient_model = nnx.merge(graph, gradients, constants)
    expected_variables = dict(_variables(reference.serialize()))
    actual_variables = dict(_variables(gradient_model.serialize()))
    assert expected_variables.keys() == actual_variables.keys()
    for name, expected_value in expected_variables.items():
        np.testing.assert_allclose(
            actual_variables[name],
            expected_value,
            rtol=3e-7,
            atol=3e-9,
            err_msg=name,
        )


@pytest.mark.parametrize("options,physics", OPTIONS[:3])
def test_symbolic_hlo_architectural_paths(
    tmp_path: Path, options: dict[str, Any], physics: bool
) -> None:
    """Attention, Cartesian and higher-degree paths retain symbolic HLO shapes."""
    config = model_config()
    config["descriptor"].update(options)
    reference = _reference(config)
    path = tmp_path / "model.hlo"
    deserialize_to_file(
        str(path), {"model": reference.serialize(), "model_def_script": config}
    )
    coord, atype, _ = _inputs()
    expected = reference(torch.tensor(coord), torch.tensor(atype))
    actual = DeepPot(str(path), auto_batch_size=False).eval(
        coord, None, atype, mixed_type=True
    )
    for value, key in zip(actual, ("energy", "force", "virial"), strict=True):
        np.testing.assert_allclose(
            value, expected[key].detach().numpy(), rtol=2e-8, atol=2e-9
        )


def _native_physics_model() -> Any:
    config = model_config(spin=True, bridging=True)
    config["fitting_net"]["vacuum_ref"] = True
    return BaseModel.deserialize(_reference(config).serialize())


def test_native_physics_symmetries_and_energy_difference() -> None:
    """Native spin and ZBL preserve Euclidean symmetries and conservative forces."""
    model = _native_physics_model()
    coord, atype, spin = _inputs()

    @nnx.jit
    def evaluate(model: Any, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return model.call_common_lower_graph(**inputs)

    def predict(
        coords: np.ndarray, types: np.ndarray, spins: np.ndarray
    ) -> dict[str, np.ndarray]:
        return jax.tree.map(
            np.asarray, evaluate(model, _graph_inputs(model, coords, types, spins))
        )

    baseline = predict(coord, atype, spin)
    shifted = predict(coord + np.array([2.4, -1.7, 0.8]), atype, spin)
    for key in (
        "energy",
        "energy_redu",
        "energy_derv_r",
        "energy_derv_c_redu",
        "energy_derv_r_mag",
    ):
        np.testing.assert_allclose(shifted[key], baseline[key], rtol=2e-9, atol=2e-10)

    permutation = np.array([2, 0, 1])
    permuted = predict(
        coord[:, permutation], atype[:, permutation], spin[:, permutation]
    )
    for key in ("energy", "energy_derv_r", "energy_derv_c", "energy_derv_r_mag"):
        np.testing.assert_allclose(
            permuted[key], baseline[key][permutation], rtol=2e-9, atol=2e-10
        )

    random = np.random.default_rng(29)
    rotation, _ = np.linalg.qr(random.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    rotated = predict(coord @ rotation.T, atype, spin @ rotation.T)
    np.testing.assert_allclose(
        rotated["energy_redu"], baseline["energy_redu"], rtol=2e-9, atol=2e-10
    )
    for key in ("energy_derv_r", "energy_derv_r_mag"):
        np.testing.assert_allclose(
            rotated[key], baseline[key] @ rotation.T, rtol=2e-8, atol=2e-9
        )
    virial = baseline["energy_derv_c_redu"][:1].reshape(3, 3)
    np.testing.assert_array_equal(baseline["energy_derv_c_redu"][1:], 0.0)
    np.testing.assert_array_equal(rotated["energy_derv_c_redu"][1:], 0.0)
    np.testing.assert_allclose(
        rotated["energy_derv_c_redu"][:1].reshape(3, 3),
        rotation @ virial @ rotation.T,
        rtol=2e-8,
        atol=2e-9,
    )

    force = baseline["energy_derv_r"].reshape(coord.shape)
    force_mag = baseline["energy_derv_r_mag"].reshape(spin.shape)
    np.testing.assert_allclose(force.sum(axis=1), 0.0, atol=1e-11)
    torque = np.cross(coord, force) + np.cross(spin, force_mag)
    np.testing.assert_allclose(torque.sum(axis=1), 0.0, atol=2e-8)

    displacement = random.normal(size=coord.shape)
    spin_displacement = random.normal(size=spin.shape)
    step = 1e-5
    positive = predict(
        coord + step * displacement, atype, spin + step * spin_displacement
    )["energy_redu"].sum()
    negative = predict(
        coord - step * displacement, atype, spin - step * spin_displacement
    )["energy_redu"].sum()
    expected_derivative = (
        -(force * displacement).sum() - (force_mag * spin_displacement).sum()
    )
    np.testing.assert_allclose(
        (positive - negative) / (2 * step), expected_derivative, rtol=2e-6, atol=2e-6
    )


def test_padded_ragged_frames_and_third_derivatives() -> None:
    """Frame, node and edge padding preserve independent-frame higher derivatives."""
    model = _native_physics_model()
    coord, atype, spin = _inputs()
    second_coord = coord[:, :2] + 0.4
    second_atype = atype[:, :2]
    second_spin = spin[:, :2]
    _, raw = prepare_graph_batch(
        {
            "coord": np.concatenate([coord[0], second_coord[0]]),
            "type": np.concatenate([atype[0], second_atype[0]]),
            "spin": np.concatenate([spin[0], second_spin[0]]),
            "n_node": np.array([3, 2]),
            "find_box": False,
        },
        rcut=model.get_rcut(),
        ntypes=model.get_ntypes(),
    )
    unpadded = jax.tree.map(jnp.asarray, raw)
    padded = deepcopy(raw)
    padded["atype"] = np.pad(padded["atype"], (0, 2), constant_values=-1)
    padded["spin"] = np.pad(padded["spin"], ((0, 2), (0, 0)))
    padded["n_node"] = np.pad(padded["n_node"], (0, 2))
    padded["edge_index"] = np.pad(padded["edge_index"], ((0, 0), (0, 4)))
    padded["edge_vec"] = np.pad(padded["edge_vec"], ((0, 4), (0, 0)))
    padded["edge_mask"] = np.pad(padded["edge_mask"], (0, 4))
    padded = jax.tree.map(jnp.asarray, padded)

    @nnx.jit
    def evaluate(model: Any, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return model.call_common_lower_graph(**inputs)

    actual = evaluate(model, padded)
    plain = evaluate(model, unpadded)
    first = evaluate(model, _graph_inputs(model, coord, atype, spin))
    second = evaluate(
        model, _graph_inputs(model, second_coord, second_atype, second_spin)
    )
    for key in ("energy", "energy_derv_r", "energy_derv_c", "energy_derv_r_mag"):
        combined = np.concatenate([first[key], second[key]])
        np.testing.assert_allclose(actual[key][:5], combined, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(plain[key], combined, rtol=2e-9, atol=2e-10)
        np.testing.assert_array_equal(actual[key][5:], 0.0)
    np.testing.assert_allclose(
        actual["energy_redu"][:2],
        np.concatenate([first["energy_redu"][:1], second["energy_redu"][:1]]),
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_array_equal(actual["energy_redu"][2:], 0.0)

    direction = jnp.asarray(
        np.random.default_rng(41).normal(0.0, 0.1, padded["edge_vec"].shape)
    )

    def energy(displacement: jax.Array) -> jax.Array:
        inputs = dict(padded)
        inputs["edge_vec"] = padded["edge_vec"] + displacement * direction
        return jnp.sum(model.call_common_lower_graph(**inputs)["energy_redu"])

    second_derivative = jax.jit(jax.grad(jax.grad(energy)))
    third_derivative = jax.jit(jax.grad(jax.grad(jax.grad(energy))))
    step = 2e-5
    actual_third = third_derivative(0.0)
    finite_difference = (second_derivative(step) - second_derivative(-step)) / (
        2 * step
    )
    assert np.isfinite(actual_third)
    assert abs(float(actual_third)) > 1e-8
    np.testing.assert_allclose(actual_third, finite_difference, rtol=3e-5, atol=2e-6)


def test_hessian_vector_norm_parameter_gradient() -> None:
    """The mixed third derivative d/dtheta(d2E/dr2) matches a parameter difference."""
    config = model_config()
    config["descriptor"].update(channels=2, n_radial=2)
    model = BaseModel.deserialize(_reference(config).serialize())
    coord, atype, _ = _inputs()
    inputs = _graph_inputs(model, coord, atype)
    positions = jnp.asarray(coord[0])
    direction = jnp.asarray(np.random.default_rng(53).normal(size=positions.shape))
    graph, parameters, constants = nnx.split(model, nnx.Param, ...)

    def hessian_vector_norm(parameters: Any) -> jax.Array:
        current = nnx.merge(graph, parameters, constants)

        def energy(positions: jax.Array) -> jax.Array:
            geometry = dict(inputs)
            geometry["edge_vec"] = (
                positions[inputs["edge_index"][0]] - positions[inputs["edge_index"][1]]
            )
            return jnp.sum(current.call_common_lower_graph(**geometry)["energy_redu"])

        product = jax.jvp(jax.grad(energy), (positions,), (direction,))[1]
        return jnp.sum(product**2)

    value, gradient = jax.jit(jax.value_and_grad(hessian_vector_norm))(parameters)
    assert np.isfinite(value) and float(value) > 1e-12
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    random = np.random.default_rng(59)
    total_size = sum(leaf.size for leaf in jax.tree.leaves(parameters))
    parameter_direction = jax.tree.map(
        lambda leaf: jnp.asarray(
            random.normal(size=leaf.shape) / np.sqrt(total_size), dtype=leaf.dtype
        ),
        parameters,
    )
    analytical = sum(
        jnp.vdot(derivative, vector)
        for derivative, vector in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(parameter_direction), strict=True
        )
    )
    assert abs(float(analytical)) > 1e-10
    step = 1e-4
    evaluate = jax.jit(hessian_vector_norm)
    positive = evaluate(
        jax.tree.map(
            lambda value, delta: value + step * delta, parameters, parameter_direction
        )
    )
    negative = evaluate(
        jax.tree.map(
            lambda value, delta: value - step * delta, parameters, parameter_direction
        )
    )
    np.testing.assert_allclose(
        analytical, (positive - negative) / (2 * step), rtol=3e-4, atol=1e-8
    )


def _evaluate(
    model: Any, coord: np.ndarray, atype: np.ndarray, steps: int = 1
) -> list[dict[str, np.ndarray]]:
    """Evaluate a model on the graph route after advancing its random state."""
    inputs = _graph_inputs(model, coord, atype)
    nframes, nnodes = atype.shape[0], atype.size

    @nnx.jit
    def run(current: Any, graph_inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return current.call_common_lower_graph(**graph_inputs)

    results = []
    for _ in range(steps):
        model.get_descriptor().advance_random_state()
        output = run(model, inputs)
        results.append(
            {
                "energy": np.asarray(output["energy_redu"], np.float64)[:nframes],
                "force": np.asarray(output["energy_derv_r"], np.float64)[:nnodes],
                "virial": np.asarray(output["energy_derv_c_redu"], np.float64)[
                    :nframes
                ],
            }
        )
    return results


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


@pytest.mark.parametrize("bridging", [False, True])
def test_random_rotation_leaves_training_outputs_unchanged(bridging: bool) -> None:
    """Rolling the edge frames at random is a gauge choice, not a model change."""
    config = model_config(bridging=bridging)
    config["descriptor"]["random_gamma"] = True
    model = BaseModel.deserialize(_reference(config).serialize())
    coord, atype, _ = _inputs()
    model.eval()
    (expected,) = _evaluate(model, coord, atype)
    assert np.linalg.norm(expected["force"]) > 1e-8
    model.train()
    key = np.asarray(model.get_descriptor().rotation_key[...]).copy()
    for step in _evaluate(model, coord, atype, steps=3):
        for name, value in expected.items():
            np.testing.assert_allclose(
                step[name], value, rtol=1e-10, atol=1e-12, err_msg=name
            )
    assert not np.array_equal(key, model.get_descriptor().rotation_key[...])


def test_amp_error_is_bfloat16_rounding_of_matrix_products() -> None:
    """AMP perturbs a float32 model at the BF16 level and not beyond it."""
    from deepmd.pt.model.model import get_model as get_pt_model

    config = model_config(bridging=True)
    reference = _reference(config)
    rounded_config = deepcopy(config)
    rounded_config["descriptor"].update(precision="float32", use_amp=True)
    rounded_config["fitting_net"]["precision"] = "float32"
    rounded = get_pt_model(rounded_config).cpu().eval()
    rounded.load_state_dict(reference.state_dict())
    exact = BaseModel.deserialize(reference.serialize())
    model = BaseModel.deserialize(rounded.serialize())
    descriptor = model.get_descriptor()
    assert descriptor.use_amp
    coord, atype, _ = _inputs()
    exact.train()
    model.train()
    (expected,) = _evaluate(exact, coord, atype)
    (with_amp,) = _evaluate(model, coord, atype)
    descriptor.use_amp = False
    (without_amp,) = _evaluate(model, coord, atype)
    assert np.linalg.norm(expected["force"]) > 1e-8
    for name, value in expected.items():
        plain = _relative_error(without_amp[name], value)
        mixed = _relative_error(with_amp[name], value)
        assert plain < 1e-5, name
        assert 100.0 * plain < mixed < 5e-2, name
