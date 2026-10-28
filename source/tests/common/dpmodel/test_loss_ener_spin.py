# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for backend-independent spin-energy losses."""

import numpy as np
import pytest

from deepmd.dpmodel.loss.ener_spin import (
    EnergySpinLoss,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("loss_func", ["mse", "mae"])
@pytest.mark.parametrize("find_force_mag", [0.0, 1.0])
def test_no_magnetic_atoms(dtype, loss_func, find_force_mag) -> None:
    """An all-false magnetic mask must not poison the total loss.

    The magnetic-force prefactor remains enabled even when the label is absent,
    reproducing the case where ``0 * NaN`` previously contaminated the loss.
    Display metrics retain the standard ``display_if_exist`` behavior: zero for
    a present-but-empty label and NaN when the label itself is absent.
    """
    nframes, natoms = 2, 6
    loss_fn = EnergySpinLoss(
        starter_learning_rate=1.0,
        start_pref_fm=1.0,
        limit_pref_fm=1.0,
        loss_func=loss_func,
    )
    model_pred = {
        # Energy selects the NumPy Array API namespace even though its loss term
        # is disabled.
        "energy": np.zeros((nframes,), dtype=dtype),
        "force_mag": np.ones((nframes, natoms, 3), dtype=dtype),
        "mask_mag": np.zeros((nframes, natoms, 1), dtype=bool),
    }
    label = {
        "force_mag": np.zeros((nframes, natoms, 3), dtype=dtype),
        "find_force_mag": find_force_mag,
    }

    loss, more_loss = loss_fn(
        1.0,
        natoms,
        model_pred,
        label,
        mae=True,
    )

    assert np.isfinite(loss)
    np.testing.assert_equal(loss, 0.0)
    np.testing.assert_equal(more_loss["rmse"], 0.0)
    metric_names = ["mae_fm"]
    if loss_func == "mse":
        metric_names.append("rmse_fm")
    for name in metric_names:
        if find_force_mag:
            np.testing.assert_equal(more_loss[name], 0.0)
        else:
            assert np.isnan(more_loss[name])
