# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 parameter conversion, derivatives and precision policies in JAX."""

from typing import (
    Any,
)

import numpy as np
import pytest

from deepmd.jax.common import (
    to_flat_state,
)
from deepmd.jax.descriptor.dpa4 import (
    DescrptDPA4,
    GatedActivation,
    _iter_object_tree,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.jax.fitting.fitting import (
    EnergyFittingNet,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


def _make_trainable_descriptor(basis_type: str = "bessel") -> DescrptDPA4:
    """Build a small descriptor that enables the optional trainable leaves."""
    return DescrptDPA4(
        ntypes=2,
        sel=4,
        rcut=4.0,
        channels=4,
        n_radial=4,
        basis_type=basis_type,
        lmax=1,
        mmax=1,
        n_blocks=1,
        grid_branch=0,
        layer_scale=True,
        message_node_so3=True,
        random_gamma=False,
        precision="float64",
        trainable=True,
        seed=20260711,
    )


def test_optional_dpa4_weights_are_jax_parameters() -> None:
    """Optional cross-grid and FFN LayerScale weights must receive gradients."""
    descriptor = _make_trainable_descriptor()
    modules = list(_iter_object_tree(descriptor))

    frame_modules = [
        module
        for module in modules
        if type(module).__name__ in {"FrameContract", "FrameExpand"}
    ]
    assert {type(module).__name__ for module in frame_modules} == {
        "FrameContract",
        "FrameExpand",
    }
    assert all(isinstance(module.weight, ArrayAPIParam) for module in frame_modules)

    interaction_blocks = [
        module for module in modules if type(module).__name__ == "SeZMInteractionBlock"
    ]
    assert interaction_blocks
    assert all(
        isinstance(scale, ArrayAPIParam)
        for block in interaction_blocks
        for scale in block.adam_ffn_layer_scales
    )


def test_moe_is_refused() -> None:
    """MoE belongs to the PyTorch Exportable backend."""
    with pytest.raises(NotImplementedError, match="pt-expt"):
        DescrptDPA4(ntypes=2, sel=4, rcut=4.0, moe={})
    for fitting in (EnergyFittingNet, SeZMEnergyFittingNet):
        with pytest.raises(NotImplementedError, match="pt-expt"):
            fitting(ntypes=2, dim_descrpt=8, mixed_types=True, moe={})


@pytest.mark.parametrize("family", ["bessel", "gaussian"])
def test_fixed_basis_is_not_an_optimizer_parameter(family: str) -> None:
    """Fixed basis arrays remain serializable without entering the parameter tree."""
    descriptor = _make_trainable_descriptor(f"{family}/fix")
    restored = DescrptDPA4.deserialize(descriptor.serialize())
    for model in (descriptor, restored):
        assert not model.radial_basis.trainable
        assert not isinstance(model.radial_basis.adam_freqs, nnx.Param)
        assert len(to_flat_state(nnx.state(model, nnx.Param))) > 0


def test_frozen_descriptor_has_no_optimizer_visible_parameters() -> None:
    """The root freeze flag must demote every descendant parameter."""
    descriptor = DescrptDPA4(
        ntypes=2,
        sel=4,
        rcut=4.0,
        channels=4,
        n_radial=4,
        lmax=1,
        mmax=1,
        n_blocks=1,
        grid_branch=0,
        random_gamma=False,
        precision="float64",
        trainable=False,
        seed=20260712,
    )

    assert len(to_flat_state(nnx.state(descriptor, nnx.Param))) == 0


def _descriptor_pair(**overrides: Any) -> tuple[Any, DescrptDPA4]:
    """Transfer nonzero residual branches from the production pt implementation."""
    torch = pytest.importorskip("torch")
    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )

    kwargs = {
        "ntypes": 2,
        "sel": 4,
        "rcut": 4.0,
        "channels": 4,
        "n_radial": 4,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 1,
        "grid_branch": 0,
        "random_gamma": False,
        "use_amp": False,
        "precision": "float64",
        "seed": 20261023,
    }
    kwargs.update(overrides)
    reference = DescrptSeZM(**kwargs).eval()
    rng = np.random.default_rng(20261023)
    with torch.no_grad():
        for param in reference.parameters():
            param.add_(
                torch.as_tensor(
                    rng.normal(0.0, 0.05, tuple(param.shape)),
                    dtype=param.dtype,
                    device=param.device,
                )
            )
    return reference, DescrptDPA4.deserialize(reference.serialize())


def _descriptor_inputs() -> list[np.ndarray]:
    """Build two frames with ghosts, invalid mappings and neighbor padding."""
    rng = np.random.default_rng(20261024)
    nf, nloc, nall, nnei = 2, 3, 5, 4
    coord = rng.uniform(0.0, 4.0, size=(nf, nall, 3))
    mapping = np.concatenate(
        [
            np.tile(np.arange(nloc, dtype=np.int64), (nf, 1)),
            rng.integers(0, nloc, size=(nf, nall - nloc)),
        ],
        axis=1,
    )
    mapping[:, -1] = -1
    atype = rng.integers(0, 2, size=(nf, nloc))
    atype_ext = np.take_along_axis(atype, np.clip(mapping, 0, nloc - 1), axis=1)
    nlist = rng.integers(0, nall, size=(nf, nloc, nnei))
    center = np.arange(nloc)[None, :, None]
    nlist = np.where(nlist == center, (nlist + 1) % nall, nlist)
    nlist[..., -1] = -1
    nlist[:, 0, 0] = nall - 1
    nlist[:, 0, 1] = 1
    return [coord, atype_ext, nlist, mapping]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"so2_norm": True},
        {"message_node_so3": True, "layer_scale": True},
        {
            "env_exp": 5,
            "basis_type": "gaussian/fix",
            "edge_norm": [False, False, True],
            "sandwich_norm": [True, False, True, False],
            "ffn_so3_grid": True,
            "so3_readout": "mlp",
        },
    ],
    ids=["default", "so2_norm", "so3_grid", "v20260911"],
)
def test_pt_forward_and_parameter_gradient_parity(options: dict[str, Any]) -> None:
    """Every trainable array has the same value and derivative as pt under JIT."""
    import torch

    reference, model = _descriptor_pair(**options)
    inputs = _descriptor_inputs()
    device = next(reference.parameters()).device
    pt_inputs = [torch.as_tensor(value, device=device) for value in inputs]
    pt_output = reference(*pt_inputs)[0]
    (pt_output**2).sum().backward()
    graph, parameters, constants = nnx.split(model, nnx.Param, ...)

    def loss(params: Any) -> tuple[jax.Array, jax.Array]:
        current = nnx.merge(graph, params, constants)
        output = current(*[jnp.asarray(value) for value in inputs])[0]
        return jnp.sum(output**2), output

    (_, output), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(parameters)
    np.testing.assert_allclose(
        output, pt_output.detach().cpu().numpy(), rtol=2e-9, atol=2e-11
    )
    assert len(jax.tree.leaves(parameters)) == sum(
        param.requires_grad for param in reference.parameters()
    )
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, name
                param.copy_(param.grad)
    gradient_model = nnx.merge(graph, gradients, constants)
    expected = reference.serialize()["@variables"]
    actual = gradient_model.serialize()["@variables"]
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(
            actual[name], value, rtol=2e-8, atol=2e-10, err_msg=name
        )


def test_coordinate_third_derivative_is_finite_and_matches_difference() -> None:
    """Nested differentiation stays valid across masked and ghost edges."""
    _, model = _descriptor_pair(lmax=1)
    coord, atype, nlist, mapping = [
        jnp.asarray(value) for value in _descriptor_inputs()
    ]
    direction = jnp.asarray(np.random.default_rng(20261025).normal(size=coord.shape))

    def energy(displacement: jax.Array) -> jax.Array:
        output = model(coord + displacement * direction, atype, nlist, mapping)[0]
        return jnp.sum(output**2)

    second = jax.jit(jax.grad(jax.grad(energy)))
    third = jax.jit(jax.grad(jax.grad(jax.grad(energy))))
    step = 2e-5
    expected = (second(step) - second(-step)) / (2 * step)
    actual = third(0.0)
    assert np.isfinite(actual)
    assert abs(float(actual)) > 1e-8
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize("layout", ["nfdc", "ndfc", "fndc"])
@pytest.mark.parametrize("explicit_gate", [False, True])
def test_vector_gate_third_derivative(layout: str, explicit_gate: bool) -> None:
    """Vector gating retains third derivatives in every supported layout."""
    layer = GatedActivation(
        lmax=2,
        mmax=1,
        channels=2,
        n_focus=2,
        precision="float64",
        layout=layout,
        mlp_bias=True,
        trainable=True,
        seed=20261029,
    )
    shape = {"nfdc": (3, 2, 7, 2), "ndfc": (3, 7, 2, 2), "fndc": (2, 3, 7, 2)}[layout]
    rng = np.random.default_rng(20261029)
    values, direction, gates, gate_direction = (
        jnp.asarray(rng.normal(size=shape)) for _ in range(4)
    )
    degree_axis = 1 if layout == "ndfc" else 2
    vectors = tuple(
        slice(1, None) if axis == degree_axis else slice(None) for axis in range(4)
    )

    def objective(displacement: jax.Array) -> jax.Array:
        gate = gates + displacement * gate_direction if explicit_gate else None
        output = layer(values + displacement * direction, gate)
        return jnp.sum(output[vectors] ** 2)

    second = jax.jit(jax.grad(jax.grad(objective)))
    third = jax.jit(jax.grad(jax.grad(jax.grad(objective))))
    step = 2e-5
    expected = (second(step) - second(-step)) / (2 * step)
    actual = third(0.0)
    assert np.isfinite(actual) and abs(float(actual)) > 1e-8
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)


def test_random_roll_state_is_reproducible_and_checkpointed() -> None:
    """A step draws fresh rolls while nested derivatives share one realization."""
    _, model = _descriptor_pair(random_gamma=True)
    model.train()
    vectors = jnp.ones((7, 3), dtype=jnp.float64)

    @nnx.jit
    def advance(current: DescrptDPA4) -> jax.Array:
        current.advance_random_state()
        return current._rotation_angles(vectors)

    first = advance(model)
    graph, state = nnx.split(model)
    restored = nnx.merge(graph, jax.tree.map(jnp.copy, state))
    second = advance(model)
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(second, advance(restored))
    before = np.asarray(model.rotation_key[...]).copy()
    model.eval()
    advance(model)
    np.testing.assert_array_equal(before, model.rotation_key[...])


def test_amp_compiles_bfloat16_products_with_float32_state() -> None:
    """Mixed precision changes matrix products while preserving master weights."""
    _, model = _descriptor_pair(precision="float32", use_amp=True, lmax=1)
    model.train()
    graph, params, constants = nnx.split(model, nnx.Param, ...)
    inputs = [jnp.asarray(value) for value in _descriptor_inputs()]

    def energy(parameters: Any) -> jax.Array:
        current = nnx.merge(graph, parameters, constants)
        return jnp.sum(current(*inputs)[0] ** 2)

    compiled = jax.jit(jax.value_and_grad(energy))
    ir = str(compiled.lower(params).compiler_ir())
    assert "bf16" in ir
    value, gradient = compiled(params)
    assert np.isfinite(value)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
