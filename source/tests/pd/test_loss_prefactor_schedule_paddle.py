# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import paddle
import pytest

from deepmd.pd.loss.ener import (
    EnergyStdLoss,
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


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("learning_rate", [1.0, 0.5, 0.0])
def test_energy_schedule_contributes_to_loss_and_gradient(start, limit, learning_rate):
    loss_fn = EnergyStdLoss(start_pref_e=start, limit_pref_e=limit)
    prediction = paddle.to_tensor([[2.0]], dtype="float64", stop_gradient=False)
    label = {"energy": paddle.zeros_like(prediction), "find_energy": 1.0}
    _, loss, _ = loss_fn({}, lambda: {"energy": prediction}, label, 2, learning_rate)
    weight = limit + (start - limit) * learning_rate
    np.testing.assert_allclose(loss.numpy(), 2.0 * weight)
    (gradient,) = paddle.grad(loss, prediction)
    np.testing.assert_allclose(gradient.numpy(), [[2.0 * weight]])


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("term", ["f", "pf", "gf"])
def test_force_schedule_contributes_to_loss_and_gradient(term, start, limit):
    loss_fn = EnergyStdLoss(
        **{f"start_pref_{term}": start, f"limit_pref_{term}": limit},
        numb_generalized_coord=1,
    )
    prediction = paddle.ones((1, 1, 3), dtype="float64")
    prediction.stop_gradient = False
    label = {
        "force": paddle.zeros_like(prediction),
        "find_force": 1.0,
        "atom_pref": paddle.ones_like(prediction),
        "find_atom_pref": 1.0,
        "drdq": paddle.to_tensor([[[1.0], [0.0], [0.0]]], dtype="float64"),
        "find_drdq": 1.0,
    }
    _, loss, _ = loss_fn({}, lambda: {"force": prediction}, label, 1, 0.25)
    weight = limit + (start - limit) * 0.25
    np.testing.assert_allclose(loss.numpy(), weight)
    expected = [2.0, 0.0, 0.0] if term == "gf" else [2.0 / 3] * 3
    (gradient,) = paddle.grad(loss, prediction)
    np.testing.assert_allclose(gradient.numpy(), np.asarray([[expected]]) * weight)
