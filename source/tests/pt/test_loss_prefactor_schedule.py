# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest
import torch

from deepmd.pt.loss.dos import (
    DOSLoss,
)
from deepmd.pt.loss.ener import (
    EnergyStdLoss,
)
from deepmd.pt.loss.ener_spin import (
    EnergySpinLoss,
)
from deepmd.pt.utils.env import (
    DEVICE,
)


@pytest.mark.parametrize(
    "start,limit", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
)
@pytest.mark.parametrize(
    "term,labels",
    [
        ("e", {"energy"}),
        ("f", {"force"}),
        ("v", {"virial"}),
        ("ae", {"atom_ener"}),
        ("pf", {"force", "atom_pref"}),
        ("gf", {"force", "drdq"}),
    ],
)
def test_energy_schedule_label_requirements(term, labels, start, limit):
    loss = EnergyStdLoss(
        **{f"start_pref_{term}": start, f"limit_pref_{term}": limit},
        numb_generalized_coord=1,
    )
    required = {item.key for item in loss.label_requirement}
    assert required == (labels if start or limit else set())


@pytest.mark.parametrize(
    "start,limit", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
)
@pytest.mark.parametrize(
    "term,label",
    [
        ("e", "energy"),
        ("fr", "force"),
        ("fm", "force_mag"),
        ("v", "virial"),
        ("ae", "atom_ener"),
    ],
)
def test_spin_schedule_label_requirements(term, label, start, limit):
    loss = EnergySpinLoss(**{f"start_pref_{term}": start, f"limit_pref_{term}": limit})
    required = {item.key for item in loss.label_requirement}
    assert required == ({label} if start or limit else set())


@pytest.mark.parametrize(
    "start,limit", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
)
@pytest.mark.parametrize("term", ["dos", "cdf", "ados", "acdf"])
def test_dos_schedule_label_requirements(term, start, limit):
    prefs = {
        f"{bound}_pref_{name}": 0.0
        for name in ("dos", "cdf", "ados", "acdf")
        for bound in ("start", "limit")
    }
    anchor = "ados" if term in ("dos", "cdf") else "dos"
    prefs[f"start_pref_{anchor}"] = prefs[f"limit_pref_{anchor}"] = 1.0
    prefs[f"start_pref_{term}"] = start
    prefs[f"limit_pref_{term}"] = limit
    loss = DOSLoss(starter_learning_rate=1.0, numb_dos=2, **prefs)
    required = {item.key for item in loss.label_requirement}
    label = "dos" if term in ("dos", "cdf") else "atom_dos"
    assert (label in required) == bool(start or limit)


@pytest.mark.parametrize("loss_type", [EnergyStdLoss, EnergySpinLoss])
@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("learning_rate", [1.0, 0.5, 0.0])
def test_energy_schedule_contributes_to_loss_and_gradient(
    loss_type, start, limit, learning_rate
):
    loss_fn = loss_type(start_pref_e=start, limit_pref_e=limit)
    prediction = torch.tensor(
        [[2.0]], dtype=torch.float64, device=DEVICE, requires_grad=True
    )
    label = {"energy": torch.zeros_like(prediction), "find_energy": 1.0}
    _, loss, _ = loss_fn({}, lambda: {"energy": prediction}, label, 2, learning_rate)
    weight = limit + (start - limit) * learning_rate
    torch.testing.assert_close(
        loss, torch.tensor(2.0 * weight, dtype=loss.dtype, device=DEVICE)
    )
    (gradient,) = torch.autograd.grad(loss, prediction)
    torch.testing.assert_close(gradient, torch.full_like(prediction, 2.0 * weight))


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("term", ["f", "pf", "gf"])
def test_force_schedule_contributes_to_loss_and_gradient(term, start, limit):
    loss_fn = EnergyStdLoss(
        **{f"start_pref_{term}": start, f"limit_pref_{term}": limit},
        numb_generalized_coord=1,
    )
    prediction = torch.ones(
        (1, 1, 3), dtype=torch.float64, device=DEVICE, requires_grad=True
    )
    label = {
        "force": torch.zeros_like(prediction),
        "find_force": 1.0,
        "atom_pref": torch.ones_like(prediction),
        "find_atom_pref": 1.0,
        "drdq": torch.tensor(
            [[[1.0], [0.0], [0.0]]], dtype=torch.float64, device=DEVICE
        ),
        "find_drdq": 1.0,
    }
    _, loss, _ = loss_fn({}, lambda: {"force": prediction}, label, 1, 0.25)
    weight = limit + (start - limit) * 0.25
    torch.testing.assert_close(
        loss, torch.tensor(weight, dtype=loss.dtype, device=DEVICE)
    )
    expected = [2.0, 0.0, 0.0] if term == "gf" else [2.0 / 3] * 3
    expected_grad = (
        torch.tensor([[expected]], dtype=prediction.dtype, device=DEVICE) * weight
    )
    (gradient,) = torch.autograd.grad(loss, prediction)
    torch.testing.assert_close(gradient, expected_grad)


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
def test_magnetic_force_schedule_contributes_to_loss_and_gradient(start, limit):
    loss_fn = EnergySpinLoss(start_pref_fm=start, limit_pref_fm=limit)
    prediction = torch.tensor(
        [[[1.0, 2.0, 3.0]]], dtype=torch.float64, device=DEVICE, requires_grad=True
    )
    mask = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    label = {"force_mag": torch.zeros_like(prediction), "find_force_mag": 1.0}
    _, loss, _ = loss_fn(
        {}, lambda: {"force_mag": prediction, "mask_mag": mask}, label, 1, 0.25
    )
    weight = limit + (start - limit) * 0.25
    torch.testing.assert_close(
        loss, torch.tensor((14.0 / 3) * weight, dtype=loss.dtype, device=DEVICE)
    )
    (gradient,) = torch.autograd.grad(loss, prediction)
    torch.testing.assert_close(gradient, prediction.detach() * (2.0 / 3) * weight)


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("learning_rate", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("term", ["dos", "cdf", "ados", "acdf"])
def test_dos_schedule_contributes_to_loss_and_gradient(
    term, start, limit, learning_rate
):
    prefs = {
        f"{bound}_pref_{name}": 0.0
        for name in ("dos", "cdf", "ados", "acdf")
        for bound in ("start", "limit")
    }
    prefs[f"start_pref_{term}"] = start
    prefs[f"limit_pref_{term}"] = limit
    loss_fn = DOSLoss(
        starter_learning_rate=1.0,
        numb_dos=2,
        **prefs,
    )
    prediction = torch.tensor(
        [[1.0, 2.0]], dtype=torch.float64, device=DEVICE, requires_grad=True
    )
    key = "atom_dos" if term in ("ados", "acdf") else "dos"
    label = {key: torch.zeros_like(prediction), f"find_{key}": 1.0}
    _, loss, _ = loss_fn({}, lambda: {key: prediction}, label, 1, learning_rate)
    weight = limit + (start - limit) * learning_rate
    cumulative = term in ("cdf", "acdf")
    torch.testing.assert_close(
        loss,
        torch.tensor(
            (5.0 if cumulative else 2.5) * weight, dtype=loss.dtype, device=DEVICE
        ),
    )
    (gradient,) = torch.autograd.grad(loss, prediction)
    expected_grad = [[4.0, 3.0]] if cumulative else [[1.0, 2.0]]
    torch.testing.assert_close(
        gradient,
        torch.tensor(expected_grad, dtype=prediction.dtype, device=DEVICE) * weight,
    )
