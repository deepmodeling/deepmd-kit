# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 training checkpoints round-trip between PyTorch and JAX."""

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

from deepmd.entrypoints.convert_backend import (
    convert_backend,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.jax.common import (
    as_pure,
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
from deepmd.jax.optimizer import (
    build_optimizer,
    update_optimizer,
)
from deepmd.jax.utils.checkpoint import (
    import_training_state,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.pt.optimizer.hybrid_muon import (
    HybridMuonOptimizer,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.checkpoint import export_training_state as export_pt_training
from deepmd.pt_expt.train.ema import (
    ModelEMA,
)
from deepmd.utils.dpa4_serialization import (
    jax_to_pt_parameter_name,
    to_pt_dpa4_model,
)

from .test_dpa4_model import (
    _reference,
    model_config,
)


def _checkpoint_optimizer(models: Any, config: dict) -> nnx.Optimizer:
    return nnx.Optimizer(models, build_optimizer(models, config), wrt=nnx.Param)


def _checkpoint(
    spin: bool,
    bridging: bool,
    lora: bool = False,
    fitting_trainable: bool = True,
) -> tuple[dict, ModelWrapper, HybridMuonOptimizer]:
    config = model_config(spin, bridging)
    config["fitting_net"]["trainable"] = fitting_trainable
    if lora:
        config["lora"] = {"rank": 2, "alpha": 3.0}
    reference = _reference(config)
    if lora:
        from deepmd.pt.model.descriptor.sezm_nn.lora import (
            apply_lora_to_sezm,
        )

        apply_lora_to_sezm(reference, rank=2, alpha=3.0)
        reference.cpu()
    wrapper = ModelWrapper(reference, model_params=config)
    optimizer = HybridMuonOptimizer(
        wrapper.parameters(),
        named_parameters=wrapper.named_parameters(),
        adam_patterns=reference.adam_route_patterns(),
        flash_muon=False,
        lr=0.0004,
    )
    average = ModelEMA(reference, decay=0.8)
    for step in range(2):
        for parameter in wrapper.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.1 + step * 0.03)
        optimizer.step()
        average.update()
    wrapper.train_infos.update(step=2, lr=0.0004)
    checkpoint = {
        "model": wrapper.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": average.state_dict(),
    }
    return checkpoint, wrapper, optimizer


@pytest.mark.parametrize("spin,bridging", [(False, False), (True, True)])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize(
    "fitting_trainable", [False, True], ids=["fitting_frozen", "fitting_trainable"]
)
def test_pt_jax_pt_checkpoint_preserves_weights_and_training_state(
    tmp_path: Path,
    spin: bool,
    bridging: bool,
    lora: bool,
    fitting_trainable: bool,
) -> None:
    checkpoint, wrapper, _ = _checkpoint(spin, bridging, lora, fitting_trainable)
    original, converted, restored = (
        tmp_path / "original.pt",
        tmp_path / "converted.jax",
        tmp_path / "restored.pt",
    )
    torch.save(checkpoint, original)
    convert_backend(INPUT=str(original), OUTPUT=str(converted))
    converted_model = BaseModel.deserialize(
        serialize_from_file(str(converted))["model"]
    )
    expected_fitting_trainable = any(
        parameter.requires_grad
        for parameter in wrapper.model["Default"].get_fitting_net().parameters()
    )
    assert (
        bool(nnx.state(converted_model.get_fitting_net(), nnx.Param))
        is expected_fitting_trainable
    )
    convert_backend(INPUT=str(converted), OUTPUT=str(restored))
    result = torch.load(restored, map_location="cpu", weights_only=True)
    for name, value in checkpoint["model"].items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(
                result["model"][name].numpy(), value.numpy(), err_msg=name
            )
    expected = export_pt_training(checkpoint, wrapper)
    actual = export_pt_training(result, wrapper)
    for name, state in expected["optimizer"]["state"].items():
        for field, value in state.items():
            np.testing.assert_array_equal(
                actual["optimizer"]["state"][name][field],
                value,
                err_msg=f"{name}.{field}",
            )
    for name, value in expected["ema"]["parameters"].items():
        np.testing.assert_array_equal(
            actual["ema"]["parameters"][name], value, err_msg=name
        )


@pytest.mark.parametrize("canonical", [False, True])
def test_bridging_options_survive_conversion_to_pt(canonical: bool) -> None:
    """Truncation and contact clamp of a JAX composition reach the SeZM wrapper."""
    from deepmd.pt.model.model.sezm_model import (
        SeZMModel,
    )
    from deepmd.utils.bridging import (
        expand_bridging_method,
    )

    config = model_config(bridging=True)
    config.update(bridging_truncation=0.8, bridging_clamp=0.6, bridging_support=2)
    model = BaseModel.deserialize(_reference(config).serialize())
    definition = expand_bridging_method(deepcopy(config)) if canonical else config
    record, pt_definition = to_pt_dpa4_model(model.serialize(), definition)
    restored = SeZMModel.deserialize(record)
    assert restored.inter_potential.truncation == pytest.approx(0.8)
    clamp = restored.atomic_model.descriptor.contact_clamp
    assert clamp.inner == pytest.approx(0.6)
    assert clamp.support == 2
    assert pt_definition["bridging_method"] == "zbl"
    assert pt_definition["bridging_truncation"] == pytest.approx(0.8)
    assert pt_definition["bridging_clamp"] == pytest.approx(0.6)
    assert pt_definition["bridging_support"] == 2


def test_converted_hybrid_muon_continues_the_pt_parameter_trajectory(
    tmp_path: Path,
) -> None:
    checkpoint, wrapper, reference_optimizer = _checkpoint(False, False)
    path = tmp_path / "original.pt"
    torch.save(checkpoint, path)
    data = serialize_from_file(str(path))
    models = {"Default": BaseModel.deserialize(data["model"])}
    optimizer = _checkpoint_optimizer(models, data["training"]["optimizer"]["config"])
    average = nnx.clone(models)
    import_training_state(models, optimizer, data["training"], average)
    gradients = jax.tree.map(
        lambda value: jnp.full_like(value, 0.21), nnx.state(models, nnx.Param)
    )
    for parameter in wrapper.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, 0.21)
    reference_optimizer.step()
    update_optimizer(optimizer, models, gradients, learning_rate=0.0004)
    expected = dict(wrapper.named_parameters())
    for path, variable in to_flat_state(nnx.state(models, nnx.Param)):
        name = "model." + jax_to_pt_parameter_name(".".join(map(str, path)))
        np.testing.assert_allclose(
            as_pure(variable),
            expected[name].detach().numpy(),
            rtol=2e-5,
            atol=2e-6,
            err_msg=name,
        )


@pytest.mark.parametrize("graph", [False, True])
def test_multitask_checkpoint_declares_branch_lower_kind(
    tmp_path: Path, graph: bool
) -> None:
    """A multi-task checkpoint declares the neighbor input of its branches."""
    from deepmd.pt.model.model import get_model as get_pt_model
    from deepmd.pt.utils.serialization import (
        serialize_from_file as serialize_pt_from_file,
    )

    config = (
        model_config()
        if graph
        else {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [4, 4],
                "rcut": 3.0,
                "rcut_smth": 2.5,
            },
            "fitting_net": {"neuron": [4]},
        }
    )
    definitions = {"model_dict": {"alpha": deepcopy(config), "zeta": deepcopy(config)}}
    models = {
        name: get_pt_model(deepcopy(config)).cpu() for name in definitions["model_dict"]
    }
    path = tmp_path / "model.pt"
    torch.save(
        {"model": ModelWrapper(models, model_params=definitions).state_dict()}, path
    )
    data = serialize_pt_from_file(str(path))
    assert data["lower_input_kind"] == ("edge_vec" if graph else "nlist")


@pytest.mark.parametrize("legacy", [False, True])
def test_multitask_conversion_preserves_shared_optimizer_and_ema(
    tmp_path: Path, legacy: bool
) -> None:
    """Storage sharing survives conversion when branch traversal orders differ."""
    config = model_config()
    definitions = {"model_dict": {"zeta": deepcopy(config), "alpha": deepcopy(config)}}
    models = {name: _reference(config) for name in definitions["model_dict"]}
    wrapper = ModelWrapper(models, model_params=definitions)
    shared = {
        "descriptor": {
            "type": "DescrptSeZM",
            "links": [
                {"model_key": name, "shared_type": "descriptor", "shared_level": 0}
                for name in ("zeta", "alpha")
            ],
        }
    }
    wrapper.share_params(shared, {"zeta": 1.0, "alpha": 1.0}, 1e-2, resume=True)
    optimizer = HybridMuonOptimizer(
        wrapper.parameters(),
        named_parameters=wrapper.named_parameters(),
        adam_patterns=models["zeta"].adam_route_patterns(),
        flash_muon=False,
    )
    average = ModelEMA(models, decay=0.8)
    for name in ("alpha", "zeta"):
        optimizer.zero_grad(set_to_none=True)
        for parameter in models[name].parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.11)
        optimizer.step()
        average.update()
    wrapper.train_infos.update(step=2, lr=0.0005)
    source = {
        "model": wrapper.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": average.state_dict(),
    }
    if legacy:
        source["model"]["_extra_state"].pop("shared_links")
    original = tmp_path / "original.pt"
    converted = tmp_path / "converted.jax"
    restored = tmp_path / "restored.pt"
    torch.save(source, original)
    convert_backend(INPUT=str(original), OUTPUT=str(converted))
    data = serialize_from_file(str(converted))
    from deepmd.jax.utils.multi_task import (
        share_jax_model_params,
    )

    jax_models = {
        name: BaseModel.deserialize(record)
        for name, record in data["model"]["model_dict"].items()
    }
    share_jax_model_params(
        jax_models, shared, model_key_prob_map=dict.fromkeys(models, 1.0), resume=True
    )
    jax_optimizer = _checkpoint_optimizer(
        jax_models, data["training"]["optimizer"]["config"]
    )
    import_training_state(
        jax_models, jax_optimizer, data["training"], nnx.clone(jax_models)
    )
    convert_backend(INPUT=str(converted), OUTPUT=str(restored))
    result = torch.load(restored, map_location="cpu", weights_only=True)
    actual_models = {name: _reference(config) for name in definitions["model_dict"]}
    actual_wrapper = ModelWrapper(actual_models, model_params=definitions)
    actual_wrapper.load_state_dict(result["model"])
    assert (
        actual_models["zeta"].get_descriptor().type_embedding
        is actual_models["alpha"].get_descriptor().type_embedding
    )
    selected = DeepPot(str(restored), head="alpha")
    torch.testing.assert_close(
        selected.deep_eval.dp.model["Default"].cpu().state_dict(),
        actual_models["alpha"].state_dict(),
    )
    selected.close()
    expected = export_pt_training(source, wrapper)
    actual = export_pt_training(result, actual_wrapper)
    assert expected["optimizer"]["state"].keys() == actual["optimizer"]["state"].keys()
    for name, state in expected["optimizer"]["state"].items():
        for field, value in state.items():
            np.testing.assert_array_equal(
                actual["optimizer"]["state"][name][field], value
            )
    for name, value in expected["ema"]["parameters"].items():
        np.testing.assert_array_equal(actual["ema"]["parameters"][name], value)
