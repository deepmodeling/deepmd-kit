# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-device loss normalization agrees with independent PyTorch ranks."""

from typing import (
    Any,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.ener_spin import (
    EnergySpinLoss,
)
from deepmd.dpmodel.loss.property import (
    PropertyLoss,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.train.distributed import (
    DistributedTraining,
)
from deepmd.pt.loss.ener import (
    EnergyStdLoss,
)
from deepmd.pt.loss.ener_spin import EnergySpinLoss as TorchSpinLoss
from deepmd.pt.loss.property import PropertyLoss as TorchPropertyLoss
from deepmd.pt.utils import (
    env,
)


def _batches(count: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rng = np.random.default_rng(253)
    batches = []
    for group in range(count):
        frames, atoms = 1 + group % 2, 2 + group
        shapes = {
            "energy": (frames, 1),
            "force": (frames, atoms, 3),
            "virial": (frames, 9),
            "atom_energy": (frames, atoms, 1),
            "force_mag": (frames, atoms, 3),
            "property": (frames, 2),
        }
        prediction = {key: rng.normal(size=shape) for key, shape in shapes.items()}
        prediction["mask"] = np.ones((frames, atoms), dtype=np.bool_)
        prediction["mask"][:, -1] = False
        prediction["mask_mag"] = np.ones((frames, atoms, 1), dtype=np.bool_)
        prediction["mask_mag"][:, 1::2] = False
        labels = {key: rng.normal(size=shape) for key, shape in shapes.items()}
        labels.update(
            {
                "find_energy": True,
                "find_force": group % 2 == 0,
                "find_virial": group % 2 != 0,
                "find_force_mag": True,
                "find_property": True,
            }
        )
        batches.append((prediction, labels))
    return batches


def _merged(
    batches: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    atomic = {"force", "force_mag", "atom_energy", "mask", "mask_mag"}
    prediction = {}
    labels = {}
    for output, position in ((prediction, 0), (labels, 1)):
        for key in batches[0][position]:
            values = [batch[position][key] for batch in batches]
            if key.startswith("find_"):
                output[key] = any(values)
                output[f"group_{key}"] = jnp.asarray(values)
            else:
                if key in atomic:
                    values = [value.reshape((-1, *value.shape[2:])) for value in values]
                output[key] = jnp.asarray(np.concatenate(values, axis=0))
    prediction["n_node"] = jnp.asarray(
        np.concatenate(
            [
                np.full(batch[0]["mask"].shape[0], batch[0]["mask"].shape[1])
                for batch in batches
            ]
        )
    )
    labels["data_group"] = jnp.asarray(
        np.concatenate(
            [
                np.full(batch[0]["energy"].shape[0], group)
                for group, batch in enumerate(batches)
            ]
        )
    )
    labels["frame_mask"] = jnp.ones_like(labels["data_group"], dtype=jnp.bool_)
    return prediction, labels


@pytest.mark.skipif(
    jax.local_device_count() < 2, reason="Two logical devices are required"
)
@pytest.mark.parametrize(
    "loss_func,force_norm", [("mse", False), ("mae", False), ("mae", True)]
)
@pytest.mark.parametrize("intensive", [False, True])
def test_grouped_energy_matches_rank_objectives(
    loss_func: str, force_norm: bool, intensive: bool
) -> None:
    options = {
        "starter_learning_rate": 0.001,
        "start_pref_e": 2.0,
        "limit_pref_e": 2.0,
        "start_pref_f": 3.0,
        "limit_pref_f": 3.0,
        "start_pref_v": 0.2,
        "limit_pref_v": 0.2,
        "loss_func": loss_func,
        "f_use_norm": force_norm,
        "intensive_ener_virial": intensive,
    }
    _assert_rank_parity(EnergyLoss(**options), EnergyStdLoss(**options))


@pytest.mark.skipif(
    jax.local_device_count() < 2, reason="Two logical devices are required"
)
@pytest.mark.parametrize("loss_func", ["mse", "mae"])
def test_grouped_native_spin_matches_rank_objectives(loss_func: str) -> None:
    options = {
        "starter_learning_rate": 0.001,
        "start_pref_e": 2.0,
        "limit_pref_e": 2.0,
        "start_pref_fr": 3.0,
        "limit_pref_fr": 3.0,
        "start_pref_fm": 1.0,
        "limit_pref_fm": 1.0,
        "start_pref_v": 0.2,
        "limit_pref_v": 0.2,
        "loss_func": loss_func,
    }
    _assert_rank_parity(EnergySpinLoss(**options), TorchSpinLoss(**options))


@pytest.mark.skipif(
    jax.local_device_count() < 2, reason="Two logical devices are required"
)
@pytest.mark.parametrize("loss_func", ["mse", "rmse"])
@pytest.mark.parametrize("intensive", [False, True])
def test_grouped_property_matches_rank_objectives(
    loss_func: str, intensive: bool
) -> None:
    options = {
        "task_dim": 2,
        "var_name": "property",
        "loss_func": loss_func,
        "intensive": intensive,
        "out_std": [1.5, 0.7],
        "out_bias": [0.2, -0.1],
    }
    _assert_rank_parity(PropertyLoss(**options), TorchPropertyLoss(**options))


def _assert_rank_parity(loss: Any, reference_loss: Any) -> None:
    distributed = DistributedTraining({})
    batches = _batches(distributed.device_count)
    prediction, labels = _merged(batches)
    fixed = {"mask", "mask_mag", "n_node"}

    def objective(parameter: Any) -> Any:
        current = {
            key: value if key in fixed else parameter * value
            for key, value in prediction.items()
        }
        return distributed.evaluate_loss(
            loss, 0.001, prediction["mask"].shape[0], current, labels
        )[0]

    actual, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.3))
    parameter = torch.tensor(
        1.3, dtype=torch.float64, device=env.DEVICE, requires_grad=True
    )
    local_losses = []
    for local_prediction, local_labels in batches:
        current = {
            key: torch.as_tensor(value, device=env.DEVICE)
            if key in fixed
            else parameter * torch.as_tensor(value, device=env.DEVICE)
            for key, value in local_prediction.items()
        }
        local_labels = {
            key: value
            if isinstance(value, bool)
            else torch.as_tensor(value, device=env.DEVICE)
            for key, value in local_labels.items()
        }
        _, value, _ = reference_loss(
            {}, lambda: current, local_labels, current["mask"].shape[1], 0.001
        )
        local_losses.append(value)
    expected = torch.stack(local_losses).mean()
    expected.backward()
    np.testing.assert_allclose(
        actual, expected.detach().cpu().numpy(), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        gradient, parameter.grad.cpu().numpy(), rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("spin", [False, True])
def test_symmetric_virial_loss_jit_gradient(spin: bool) -> None:
    """Compiled loss projects gradients without changing the virial input."""
    options = {
        "starter_learning_rate": 1.0,
        "start_pref_e": 0.0,
        "limit_pref_e": 0.0,
        "start_pref_v": 1.0,
        "limit_pref_v": 1.0,
        "symm_virial": True,
    }
    if not spin:
        options.update(start_pref_f=0.0, limit_pref_f=0.0)
    loss = (EnergySpinLoss if spin else EnergyLoss)(**options)
    virial = jnp.arange(9, dtype=jnp.float64).reshape(1, 9)
    labels = {"virial": jnp.zeros_like(virial), "find_virial": 1.0}

    def objective(value: Any) -> Any:
        prediction = {"energy": jnp.zeros((1, 1)), "virial": value}
        return loss(1.0, 2, prediction, labels)[0]

    value, gradient = jax.jit(jax.value_and_grad(objective))(virial)
    symmetric = (
        np.asarray(virial).reshape(3, 3) + np.asarray(virial).reshape(3, 3).T
    ) / 2
    np.testing.assert_allclose(value, np.mean(symmetric**2) / 2)
    np.testing.assert_allclose(gradient.reshape(3, 3), symmetric / 9)
    np.testing.assert_array_equal(virial, np.arange(9).reshape(1, 9))
