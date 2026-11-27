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
    """Request the labels needed whenever a scheduled term is active."""
    loss = EnergyStdLoss(
        **{f"start_pref_{term}": start, f"limit_pref_{term}": limit},
        numb_generalized_coord=1,
    )
    required = {item.key for item in loss.label_requirement}
    assert required == (labels if start or limit else set())


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("learning_rate", [1.0, 0.5, 0.0])
def test_energy_schedule_contributes_to_loss_and_gradient(start, limit, learning_rate):
    """Apply the interpolated energy weight to the loss and gradient."""
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
    """Apply force, prefactor-force and generalized-force schedules."""
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


@pytest.mark.parametrize("term", ["pf", "gf"])
def test_force_terms_ignore_missing_force_labels(term):
    """Do not train force terms against a missing-label placeholder."""
    loss_fn = EnergyStdLoss(
        **{f"start_pref_{term}": 1.0, f"limit_pref_{term}": 1.0},
        numb_generalized_coord=1,
    )
    prediction = paddle.ones((1, 1, 3), dtype="float64")
    prediction.stop_gradient = False
    label = {
        "force": paddle.zeros_like(prediction),
        "find_force": 0.0,
        "atom_pref": paddle.ones_like(prediction),
        "find_atom_pref": 1.0,
        "drdq": paddle.to_tensor([[[1.0], [0.0], [0.0]]], dtype="float64"),
        "find_drdq": 1.0,
    }
    _, loss, metrics = loss_fn({}, lambda: {"force": prediction}, label, 1, 0.25)
    np.testing.assert_allclose(loss.numpy(), 0.0)
    (gradient,) = paddle.grad(loss, prediction)
    np.testing.assert_allclose(gradient.numpy(), 0.0)
    assert np.isnan(metrics[f"rmse_{term}"].numpy())


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
@pytest.mark.parametrize("term,key,width", [("e", "energy", 1), ("v", "virial", 9)])
def test_mae_extensive_terms_are_normalized(term, key, width, start, limit):
    """Normalize MAE energy and virial over the batch and atoms."""
    loss_fn = EnergyStdLoss(
        **{f"start_pref_{term}": start, f"limit_pref_{term}": limit}, loss_func="mae"
    )
    prediction = paddle.to_tensor([[2.0] * width, [6.0] * width], dtype="float64")
    prediction.stop_gradient = False
    label = {key: paddle.zeros_like(prediction), f"find_{key}": 1.0}
    _, loss, metrics = loss_fn({}, lambda: {key: prediction}, label, 2, 0.25)
    weight = limit + (start - limit) * 0.25
    np.testing.assert_allclose(loss.numpy(), 2.0 * weight)
    np.testing.assert_allclose(metrics[f"mae_{term}"].numpy(), 2.0)
    (gradient,) = paddle.grad(loss, prediction)
    np.testing.assert_allclose(gradient.numpy(), weight / (4 * width))


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0)])
def test_mae_force_uses_relative_error(start, limit):
    """Apply relative normalization before computing force MAE."""
    loss_fn = EnergyStdLoss(
        start_pref_f=start, limit_pref_f=limit, relative_f=1.0, loss_func="mae"
    )
    prediction = paddle.to_tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype="float64")
    prediction.stop_gradient = False
    label = {
        "force": paddle.to_tensor(
            [[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]], dtype="float64"
        ),
        "find_force": 1.0,
    }
    _, loss, metrics = loss_fn({}, lambda: {"force": prediction}, label, 2, 0.25)
    weight = limit + (start - limit) * 0.25
    np.testing.assert_allclose(loss.numpy(), 25 / 36 * weight)
    np.testing.assert_allclose(metrics["mae_f"].numpy(), 25 / 36)
    (gradient,) = paddle.grad(loss, prediction)
    expected = [[[-1 / 36, -1 / 36, 0.0], [1 / 6, 1 / 6, 1 / 6]]]
    np.testing.assert_allclose(gradient.numpy(), np.asarray(expected) * weight)


@pytest.mark.parametrize("start,limit", [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
def test_virial_huber_takes_precedence_over_mae(start, limit):
    """Keep the configured Huber virial loss when MAE is selected."""
    loss_fn = EnergyStdLoss(
        start_pref_v=start,
        limit_pref_v=limit,
        loss_func="mae",
        use_huber=True,
        huber_delta=1.0,
    )
    prediction = paddle.full((1, 9), 4.0, dtype="float64")
    prediction.stop_gradient = False
    label = {"virial": paddle.zeros_like(prediction), "find_virial": 1.0}
    _, loss, _ = loss_fn({}, lambda: {"virial": prediction}, label, 2, 0.25)
    weight = limit + (start - limit) * 0.25
    np.testing.assert_allclose(loss.numpy(), 1.5 * weight)
    (gradient,) = paddle.grad(loss, prediction)
    np.testing.assert_allclose(gradient.numpy(), weight / 18)
