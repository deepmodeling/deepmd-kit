# SPDX-License-Identifier: LGPL-3.0-or-later
"""Case-conditioned DPA4 fitting values and gradients against PyTorch."""

import numpy as np
import optax
import pytest
import torch

from deepmd.jax.common import (
    as_pure,
    to_flat_state,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt.model.task.sezm_ener import SeZMEnergyFittingNet as PTFitting


@pytest.mark.parametrize(
    "trainable",
    [False, True, [False, False], [True, True], [True, False], [False, True]],
)
@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("source", ["constructor", "roundtrip", "legacy", "pytorch"])
def test_fitting_trainability_survives_optimizer_and_serialization(
    trainable: bool | list[bool],
    film: bool,
    source: str,
) -> None:
    """SGD updates only the fitting weights selected by the PT freeze contract."""
    configuration = {
        "ntypes": 2,
        "dim_descrpt": 2,
        "neuron": [3],
        "precision": "float64",
        "seed": 17,
        "trainable": trainable,
        "dim_case_embd": 2 if film else 0,
        "case_film_embd": film,
    }
    reference = PTFitting(**configuration).cpu()
    expected_count = sum(
        parameter.requires_grad for parameter in reference.parameters()
    )
    model = SeZMEnergyFittingNet(**configuration)
    if film:
        reference.set_case_embd(1)
        model.set_case_embd(1)
    if source != "constructor":
        record = reference.serialize() if source == "pytorch" else model.serialize()
        if source == "legacy":
            for network in record["nets"]["networks"]:
                network.pop("trainable", None)
        model = SeZMEnergyFittingNet.deserialize(record)
        if source == "legacy":
            assert all(
                "trainable" not in network for network in record["nets"]["networks"]
            )
    descriptor = jnp.asarray([[[0.2, -0.3], [0.4, 0.5]]])
    atype = jnp.asarray([[0, 1]])
    before = {
        path: np.asarray(as_pure(variable)).copy()
        for path, variable in to_flat_state(nnx.state(model))
    }

    def loss(current: SeZMEnergyFittingNet) -> jax.Array:
        return jnp.sum((current(descriptor, atype)["energy"] - 0.5) ** 2)

    gradient = nnx.grad(loss)(model)
    parameters = nnx.state(model, nnx.Param)
    optimizer = optax.sgd(learning_rate=0.1)
    updates, _ = optimizer.update(gradient, optimizer.init(parameters), parameters)
    nnx.update(model, optax.apply_updates(parameters, updates))
    changed = [
        path
        for path, variable in to_flat_state(nnx.state(model))
        if not np.array_equal(as_pure(variable), before[path])
    ]
    assert bool(changed) is bool(expected_count), f"Unexpected SGD changes: {changed}"
    assert len(to_flat_state(nnx.state(model, nnx.Param))) == expected_count
    input_gradient = jax.grad(lambda value: jnp.sum(model(value, atype)["energy"]))(
        descriptor
    )
    assert np.isfinite(input_gradient).all()
    assert np.linalg.norm(input_gradient) > 0.0


@pytest.mark.parametrize("backend", ["pytorch", "pytorch_expt"])
@pytest.mark.parametrize(
    "trainable",
    [False, True, [False, False], [True, True], [True, False], [False, True]],
)
@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("source", ["constructor", "roundtrip", "legacy"])
def test_torch_fitting_freeze_contract_survives_sgd(
    backend: str,
    trainable: bool | list[bool],
    film: bool,
    source: str,
) -> None:
    """Torch-backed fitting retains configured freezing through real SGD steps."""
    from deepmd.pt_expt.fitting.dpa4_ener import SeZMEnergyFittingNet as ExportFitting

    fitting_type = PTFitting if backend == "pytorch" else ExportFitting
    model = fitting_type(
        ntypes=2,
        dim_descrpt=2,
        neuron=[3],
        trainable=trainable,
        precision="float64",
        seed=17,
        dim_case_embd=2 if film else 0,
        case_film_embd=film,
    )
    if film:
        model.set_case_embd(1)
    if source != "constructor":
        record = model.serialize()
        if source == "legacy":
            for network in record["nets"]["networks"]:
                network.pop("trainable", None)
        model = fitting_type.deserialize(record)
        if source == "legacy":
            assert all(
                "trainable" not in network for network in record["nets"]["networks"]
            )
    model.cpu()
    expected_trainable = all(trainable) if isinstance(trainable, list) else trainable
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    descriptor = torch.tensor(
        [[[0.2, -0.3], [0.4, 0.5]]],
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )
    atype = torch.tensor([[0, 1]], device="cpu")
    optimizer = torch.optim.SGD([descriptor, *model.parameters()], lr=0.1)
    loss = (model(descriptor, atype)["energy"] - 0.5).square().sum()
    loss.backward()
    optimizer.step()
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before[name])
    ]
    assert bool(changed) is expected_trainable, f"Unexpected SGD changes: {changed}"
    assert all(
        value.requires_grad is expected_trainable for value in model.parameters()
    )
    assert descriptor.grad is not None and torch.isfinite(descriptor.grad).all()


@pytest.mark.parametrize("film", [False, True])
@pytest.mark.parametrize("vacuum", [False, True])
@pytest.mark.parametrize("conditioning", [False, True])
def test_case_conditioned_fitting_parameter_gradients(
    film: bool,
    vacuum: bool,
    conditioning: bool,
) -> None:
    """Every GLU and case-conditioner parameter differentiates the same energy."""
    reference = PTFitting(
        ntypes=2,
        dim_descrpt=4,
        neuron=[6, 4],
        dim_case_embd=2,
        case_film_embd=film,
        numb_fparam=1 if conditioning else 0,
        numb_aparam=2 if conditioning else 0,
        precision="float64",
        vacuum_ref=vacuum,
        seed=17,
    )
    reference.set_case_embd(1)
    reference.cpu()
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(
                0.1
                * torch.randn(
                    parameter.shape, dtype=parameter.dtype, generator=generator
                )
            )
    model = SeZMEnergyFittingNet.deserialize(reference.serialize())
    random = np.random.default_rng(13)
    descriptor = random.normal(size=(2, 3, 4))
    vacuum_descriptor = random.normal(size=(2, 4)) if vacuum else None
    fparam = random.normal(size=(2, 1)) if conditioning else None
    aparam = random.normal(size=(2, 3, 2)) if conditioning else None
    atype = np.array([[0, 1, 0], [1, 0, 1]])
    source = torch.tensor(descriptor, requires_grad=True)
    expected = reference(
        source,
        torch.tensor(atype),
        fparam=torch.tensor(fparam) if conditioning else None,
        aparam=torch.tensor(aparam) if conditioning else None,
        vacuum_descriptor=torch.tensor(vacuum_descriptor) if vacuum else None,
    )["energy"]
    expected.square().sum().backward()

    def loss(model: SeZMEnergyFittingNet, descriptor: jax.Array) -> jax.Array:
        value = model(
            descriptor,
            jnp.asarray(atype),
            fparam=jnp.asarray(fparam) if conditioning else None,
            aparam=jnp.asarray(aparam) if conditioning else None,
            vacuum_descriptor=jnp.asarray(vacuum_descriptor) if vacuum else None,
        )["energy"]
        return jnp.sum(value**2)

    value, gradients = nnx.jit(nnx.value_and_grad(loss, argnums=(0, 1)))(
        model, jnp.asarray(descriptor)
    )
    np.testing.assert_allclose(
        value, expected.detach().square().sum().numpy(), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        gradients[1], source.grad.numpy(), rtol=1e-10, atol=1e-12
    )
    expected_gradients = {
        name: parameter.grad.numpy() for name, parameter in reference.named_parameters()
    }
    checked = set()
    for path, variable in to_flat_state(gradients[0]):
        name = ".".join(str(part) for part in path).replace(
            "nets._networks", "filter_layers.networks"
        )
        suffix = path[-1]
        if suffix in ("w", "b"):
            field = "matrix" if suffix == "w" else "bias"
            if ".hidden_layers." in name:
                field = "linear." + field
            name = name.rsplit(".", 1)[0] + "." + field
        np.testing.assert_allclose(
            as_pure(variable),
            expected_gradients[name],
            rtol=1e-9,
            atol=1e-11,
            err_msg=name,
        )
        checked.add(name)
    assert checked == set(expected_gradients)
