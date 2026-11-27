# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 LoRA forward, derivative, freezing, sharing, and checkpoint contracts."""

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np
import optax
import pytest
import torch

from deepmd.jax.common import (
    as_pure,
    set_variable_value,
    to_flat_state,
)
from deepmd.jax.descriptor.lora import (
    LoRASO2,
    LoRASO3,
    apply_lora_to_sezm,
    has_lora,
    merge_lora_into_base,
    serialize_merged_model,
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
    build_hybrid_muon,
    update_optimizer,
)
from deepmd.jax.optimizer.hybrid_muon import (
    MuonState,
    _parameter_name,
)
from deepmd.jax.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pt.model.descriptor.sezm_nn.lora import LoRASO2 as LoRASO2PT
from deepmd.pt.model.descriptor.sezm_nn.lora import LoRASO3 as LoRASO3PT
from deepmd.pt.model.descriptor.sezm_nn.lora import apply_lora_to_sezm as apply_lora_pt
from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Linear as SO2LinearPT
from deepmd.pt.model.descriptor.sezm_nn.so3 import SO3Linear as SO3LinearPT
from deepmd.pt.model.model import get_model as get_pt_model
from deepmd.utils.argcheck import (
    validate_no_multitask_lora,
)


def _arrays(module: Any, filter_type: type = nnx.Variable) -> dict[str, np.ndarray]:
    return {
        ".".join(str(part) for part in path): np.asarray(as_pure(value))
        for path, value in nnx.iter_graph(module)
        if isinstance(value, filter_type)
    }


def _module_pair(
    kind: str, n_focus: int, *, zero_delta: bool = False
) -> tuple[Any, Any]:
    config = {
        "lmax": 2,
        "in_channels": 3,
        "out_channels": 4,
        "n_focus": n_focus,
        "mlp_bias": True,
        "seed": 41,
        "trainable": True,
    }
    jax_class, pt_class, pt_base = (
        (LoRASO3, LoRASO3PT, SO3LinearPT)
        if kind == "so3"
        else (LoRASO2, LoRASO2PT, SO2LinearPT)
    )
    if kind == "so2":
        config["mmax"] = 1
    module = jax_class(**config, precision="float64", lora_rank=2, lora_alpha=3.0)
    reference = pt_class(
        pt_base(**config, dtype=torch.float64), rank=2, alpha=3.0
    ).cpu()
    rng = np.random.default_rng(67)
    for path, value in nnx.iter_graph(module):
        if isinstance(value, nnx.Param) and any(
            str(part).startswith("B_") for part in path
        ):
            if not zero_delta:
                set_variable_value(
                    value, jnp.asarray(rng.normal(size=value.shape) * 0.2)
                )
    arrays = _arrays(module)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            parameter.copy_(torch.tensor(arrays[name]))
            if "bias" in name:
                parameter.requires_grad_(True)
    return module, reference


@pytest.mark.parametrize("kind", ["so2", "so3"])
@pytest.mark.parametrize("n_focus", [1, 2])
def test_adapters_match_torch_values_and_gradients(kind: str, n_focus: int) -> None:
    module, reference = _module_pair(kind, n_focus)
    rng = np.random.default_rng(91)
    shape = (3, 9, n_focus, 3) if kind == "so3" else (n_focus, 3, module.reduced_dim, 3)
    inputs = rng.normal(size=shape)
    tensor = torch.tensor(inputs, requires_grad=True)
    expected = reference(tensor)
    actual = nnx.jit(lambda mod, values: mod(values))(module, jnp.asarray(inputs))
    np.testing.assert_allclose(
        actual, expected.detach().numpy(), atol=2e-12, rtol=2e-12
    )
    expected.square().sum().backward()
    gradients = nnx.jit(nnx.grad(lambda mod: jnp.sum(mod(jnp.asarray(inputs)) ** 2)))(
        module
    )
    for path, variable in to_flat_state(gradients):
        name = ".".join(str(part) for part in path)
        parameter = dict(reference.named_parameters())[name]
        np.testing.assert_allclose(
            as_pure(variable), parameter.grad.numpy(), atol=2e-11, rtol=2e-11
        )
    input_grad = jax.jit(jax.grad(lambda value: jnp.sum(module(value) ** 2)))(
        jnp.asarray(inputs)
    )
    np.testing.assert_allclose(input_grad, tensor.grad.numpy(), atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("n_focus", [1, 2])
def test_scalar_projection_keeps_adapter_and_merge(n_focus: int) -> None:
    module, reference = _module_pair("so3", n_focus)
    inputs = jnp.asarray(np.random.default_rng(23).normal(size=(2, 9, n_focus, 3)))
    np.testing.assert_allclose(
        module.call_scalar(inputs), module(inputs)[:, :1], atol=1e-12
    )
    merged = module.merge_into_base()
    np.testing.assert_allclose(
        module.call_scalar(inputs), merged.call_scalar(np.asarray(inputs)), atol=1e-12
    )
    tensor = torch.tensor(np.asarray(inputs), requires_grad=True)
    scalar = reference.forward_scalar(tensor)
    np.testing.assert_allclose(
        scalar.detach().numpy(), module.call_scalar(inputs), atol=1e-12
    )
    torch.testing.assert_close(scalar, reference(tensor)[:, :1])
    torch.testing.assert_close(
        scalar, reference.merge_into_base().cpu().forward_scalar(tensor)
    )
    scalar.sum().backward()
    assert reference.B_by_l.grad is not None
    assert torch.count_nonzero(reference.B_by_l.grad[0]) > 0
    assert torch.count_nonzero(reference.B_by_l.grad[1:]) == 0


@pytest.mark.parametrize("kind", ["so2", "so3"])
def test_zero_delta_unlock_and_hybrid_muon(kind: str) -> None:
    module, _ = _module_pair(kind, 2, zero_delta=True)
    shape = (3, 9, 2, 3) if kind == "so3" else (2, 3, module.reduced_dim, 3)
    inputs = jnp.asarray(np.random.default_rng(42).normal(size=shape))

    def objective(mod: Any) -> jax.Array:
        return jnp.sum(mod(inputs) ** 2)

    gradients = nnx.grad(objective)(module)
    for path, value in to_flat_state(gradients):
        if any(str(part).startswith("A_") for part in path):
            np.testing.assert_array_equal(as_pure(value), 0.0)
        elif any(str(part).startswith("B_") for part in path):
            assert np.linalg.norm(as_pure(value)) > 0
    params = nnx.state(module, nnx.Param)
    transform = build_hybrid_muon()
    state = transform.init(params)
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    for (path, _), current in zip(flat, state.parameters, strict=True):
        if any(
            segment.startswith(("A_", "B_"))
            for segment in _parameter_name(path).split(".")
        ):
            assert isinstance(current, MuonState)
    changes, _ = jax.jit(transform.update)(
        gradients, state, params, learning_rate=0.001
    )
    nnx.update(module, optax.apply_updates(params, changes))
    unlocked = nnx.grad(objective)(module)
    for path, value in to_flat_state(unlocked):
        if any(str(part).startswith("A_") for part in path):
            assert np.linalg.norm(as_pure(value)) > 0


def _model_config() -> dict[str, Any]:
    return {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "sel": 8,
            "rcut": 4.0,
            "channels": 4,
            "n_radial": 4,
            "lmax": 1,
            "n_blocks": 1,
            "grid_branch": 0,
            "random_gamma": False,
            "precision": "float64",
            "mlp_bias": True,
            "seed": 31,
        },
        "fitting_net": {"neuron": [4], "precision": "float64", "seed": 32},
    }


def test_model_freezing_checkpoint_and_merge() -> None:
    config = _model_config()
    plain = get_model(config)
    config["lora"] = {"rank": 2, "alpha": 3.0}
    model = get_model(config)
    assert has_lora(model)
    repeated = _arrays(get_model(config), nnx.Param)
    for name, value in _arrays(model, nnx.Param).items():
        np.testing.assert_array_equal(value, repeated[name])
    descriptor = model.atomic_model.descriptor
    assert not isinstance(descriptor.type_embedding.adam_type_embedding, nnx.Param)
    assert not isinstance(descriptor.radial_basis.adam_freqs, nnx.Param)
    assert isinstance(descriptor.radial_embedding.net[0].w, nnx.Param)
    assert isinstance(descriptor.env_seed_embedding.rbf_proj_layer1.w, nnx.Param)
    arrays = _arrays(model, nnx.Param)
    assert all("gate_linear" not in name for name in arrays)
    for _, module in nnx.iter_graph(model):
        if isinstance(module, LoRASO3):
            assert not isinstance(module.weight, nnx.Param)
            assert isinstance(module.bias, nnx.Param)
    before = {name: value.copy() for name, value in arrays.items()}
    apply_lora_to_sezm(model, rank=2, alpha=3.0)
    assert before.keys() == _arrays(model, nnx.Param).keys()
    with pytest.raises(ValueError, match="Merge the existing adapters"):
        apply_lora_to_sezm(model, rank=3, alpha=3.0)
    for name, value in _arrays(model, nnx.Param).items():
        np.testing.assert_array_equal(value, before[name])
    coord = jnp.array([[[0.0, 0.0, 0.0], [0.9, 0.2, -0.1], [-0.2, 0.8, 0.1]]])
    atype = jnp.array([[0, 1, 1]])
    expected = plain(coord, atype)
    initial = model(coord, atype)
    for key in ("energy", "force", "virial"):
        np.testing.assert_allclose(initial[key], expected[key], atol=2e-10, rtol=2e-10)
    for path, value in nnx.iter_graph(model):
        if isinstance(value, nnx.Param) and any(
            str(part).startswith("B_") for part in path
        ):
            set_variable_value(value, jnp.full_like(as_pure(value), 0.03))
    reference = model(coord, atype)
    restored = BaseModel.deserialize(model.serialize())
    assert has_lora(restored)
    assert _arrays(restored, nnx.Param).keys() == _arrays(model, nnx.Param).keys()
    merged = BaseModel.deserialize(serialize_merged_model(model))
    assert not has_lora(merged)
    assert has_lora(model)
    for candidate in (restored, merged):
        prediction = candidate(coord, atype)
        for key in ("energy", "force", "virial"):
            np.testing.assert_allclose(
                prediction[key], reference[key], atol=2e-10, rtol=2e-10
            )


def test_shared_adapters_retain_identity() -> None:
    left = get_model(_model_config())
    right = get_model(_model_config())
    right.atomic_model.descriptor = left.atomic_model.descriptor
    models = nnx.Dict(left=left, right=right)
    apply_lora_to_sezm(models, rank=2)
    assert left.atomic_model.descriptor is right.atomic_model.descriptor
    assert has_lora(left) and has_lora(right)
    merge_lora_into_base(models)
    assert left.atomic_model.descriptor is right.atomic_model.descriptor
    assert not has_lora(left) and not has_lora(right)


def test_trainable_policy_matches_torch_and_frozen_base() -> None:
    config = _model_config()
    config["lora"] = {"rank": 2}
    model = get_model(config)
    reference = get_pt_model(config)
    apply_lora_pt(reference, rank=2)
    prefix = "atomic_model.descriptor."
    jax_names = set()
    for name in _arrays(model, nnx.Param):
        if name.startswith(prefix):
            parts = name.split(".")
            parts[-1] = {"w": "matrix", "b": "bias"}.get(parts[-1], parts[-1])
            jax_names.add(".".join(parts))
    torch_names = {
        name
        for name, value in reference.named_parameters()
        if value.requires_grad and name.startswith(prefix)
    }
    assert jax_names == torch_names
    config["descriptor"]["trainable"] = False
    config["fitting_net"]["trainable"] = False
    frozen_base = get_model(config)
    assert _arrays(frozen_base, nnx.Param).keys() == _arrays(model, nnx.Param).keys()


@pytest.mark.parametrize("film", [False, True])
def test_lora_fitting_unfreeze_survives_model_serialization(film: bool) -> None:
    """The whole-fitting LoRA override includes frozen FiLM parameters."""
    from deepmd.utils.dpa4_serialization import (
        jax_to_pt_parameter_name,
    )

    config = _model_config()
    config["fitting_net"].update(
        trainable=False,
        dim_case_embd=2 if film else 0,
        case_film_embd=film,
    )
    config["lora"] = {"rank": 2}
    reference = get_pt_model(config)
    apply_lora_pt(reference, rank=2)
    expected = {
        name
        for name, value in reference.named_parameters()
        if name.startswith("atomic_model.fitting_net.") and value.requires_grad
    }
    model = get_model(config)
    candidates = (
        model,
        BaseModel.deserialize(model.serialize()),
        BaseModel.deserialize(reference.serialize()),
    )
    for candidate in candidates:
        actual = {
            jax_to_pt_parameter_name(name)
            for name in _arrays(candidate, nnx.Param)
            if name.startswith("atomic_model.fitting_net.")
        }
        assert actual == expected


def test_force_loss_trains_adapters_and_preserves_frozen_weights() -> None:
    config = _model_config()
    config["descriptor"]["ffn_neurons"] = 4
    config["lora"] = {"rank": 2, "alpha": 3.0}
    model = get_model(config)
    coord = jnp.array([[[0.0, 0.0, 0.0], [0.9, 0.2, -0.1], [-0.2, 0.8, 0.1]]])
    atype = jnp.array([[0, 1, 1]])
    neighbors = jnp.array([[[1, 2], [0, 2], [0, 1]]])
    mapping = jnp.array([[0, 1, 2]])

    def loss(mod: Any) -> jax.Array:
        prediction = mod.call_lower(coord, atype, neighbors, mapping=mapping)
        return jnp.sum((prediction["extended_force"] - 0.1) ** 2) + jnp.sum(
            (prediction["energy"] - 0.2) ** 2
        )

    gradient = nnx.jit(nnx.grad(loss))
    initial = _arrays(model)
    optimizer = nnx.Optimizer(
        model,
        build_hybrid_muon(adam_patterns=model.adam_route_patterns()),
        wrt=nnx.Param,
    )
    for _ in range(2):
        grads = gradient(model)
        for value in jax.tree_util.tree_leaves(grads):
            assert np.all(np.isfinite(value))
        update_optimizer(optimizer, model, grads, learning_rate=0.001)
    final = _arrays(model)
    frozen = {name for name in initial if name not in _arrays(model, nnx.Param)}
    for name in frozen:
        np.testing.assert_array_equal(final[name], initial[name])
    adapter_changes = {
        name: np.linalg.norm(final[name] - initial[name])
        for name in initial
        if name.endswith(("A_by_l", "B_by_l"))
    }
    assert any(
        change > 0
        for name, change in adapter_changes.items()
        if name.endswith("A_by_l")
    )
    assert any(
        change > 0
        for name, change in adapter_changes.items()
        if name.endswith("B_by_l")
    )


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("bridging", [False, True])
def test_torch_lora_model_conversion(spin: bool, bridging: bool) -> None:
    """Adapters survive ordinary, native-spin, and analytical model compositions."""
    config = _model_config()
    config["lora"] = {"rank": 2, "alpha": 3.0}
    if spin:
        config["spin"] = {"scheme": "native", "use_spin": [True, False]}
    if bridging:
        config["bridging_method"] = "zbl"
    reference = get_pt_model(config).cpu().eval()
    apply_lora_pt(reference, rank=2, alpha=3.0)
    reference.cpu()
    generator = torch.Generator().manual_seed(79)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(
                0.02
                * torch.randn(
                    parameter.shape, dtype=parameter.dtype, generator=generator
                )
            )
    model = BaseModel.deserialize(reference.serialize())
    assert has_lora(model)
    coord = np.array([[[0.0, 0.0, 0.0], [0.9, 0.2, -0.1], [-0.2, 0.8, 0.1]]])
    atype = np.array([[0, 1, 0]])
    spins = np.array([[[0.2, 0.1, 0.3], [0.0, 0.0, 0.0], [-0.1, 0.3, 0.2]]])
    kwargs = {"spin": torch.tensor(spins)} if spin else {}
    expected = reference(torch.tensor(coord), torch.tensor(atype), **kwargs)
    kwargs = {"spin": jnp.asarray(spins)} if spin else {}
    restored = BaseModel.deserialize(model.serialize())
    merged = BaseModel.deserialize(serialize_merged_model(model))
    assert has_lora(restored)
    assert not has_lora(merged)
    for candidate in (model, restored, merged):
        actual = candidate(jnp.asarray(coord), jnp.asarray(atype), **kwargs)
        for key in ("energy", "force", "virial", *(["force_mag"] if spin else [])):
            np.testing.assert_allclose(
                actual[key],
                expected[key].detach().numpy(),
                rtol=2e-9,
                atol=2e-10,
                err_msg=f"{key}, spin={spin}, bridging={bridging}",
            )


@pytest.mark.parametrize("backend", [None, "pt", "pt_expt", "tf", "jax"])
def test_multitask_lora_backend_boundary(backend: str | None) -> None:
    data = {"model": {"model_dict": {"task": {"lora": {"rank": 2}}}}}
    if backend == "jax":
        validate_no_multitask_lora(data, multi_task=True, backend=backend)
    else:
        with pytest.raises(ValueError, match="single-task"):
            validate_no_multitask_lora(data, multi_task=True, backend=backend)


def test_shared_lora_config_validation() -> None:
    config = _model_config()
    branch = {**config, "descriptor": "shared", "lora": {"rank": 2}}
    multi = {
        "shared_dict": {"shared": config["descriptor"]},
        "model_dict": {"a": deepcopy(branch), "b": deepcopy(branch)},
    }
    preprocess_shared_params(deepcopy(multi))
    incompatible = deepcopy(multi)
    incompatible["model_dict"]["b"]["lora"] = {"rank": 3}
    with pytest.raises(ValueError, match="identical LoRA rank and alpha"):
        preprocess_shared_params(incompatible)
    incompatible = deepcopy(multi)
    del incompatible["model_dict"]["b"]["lora"]
    with pytest.raises(ValueError, match="same LoRA freezing policy"):
        preprocess_shared_params(incompatible)
    partial = deepcopy(multi)
    partial["model_dict"]["b"]["descriptor"] = "shared:1"
    partial["model_dict"]["b"]["lora"] = {"rank": 3}
    preprocess_shared_params(partial)
