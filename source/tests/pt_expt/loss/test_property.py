# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the pt_expt PropertyLoss wrapper."""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.property import PropertyLoss as PropertyLossDP
from deepmd.pt_expt.loss.property import (
    PropertyLoss,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...pt.model.test_mlp import (
    get_tols,
)
from ...seed import (
    GLOBAL_SEED,
)


def _make_data(
    rng: np.random.Generator,
    nframes: int,
    task_dim: int,
    var_name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build model prediction and label dicts as torch tensors."""
    model_pred = {
        var_name: torch.tensor(
            rng.random((nframes, task_dim)), dtype=dtype, device=device
        ),
    }
    label = {
        var_name: torch.tensor(
            rng.random((nframes, task_dim)), dtype=dtype, device=device
        ),
    }
    return model_pred, label


class TestPropertyLoss:
    def setup_method(self) -> None:
        self.device = env.DEVICE

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    @pytest.mark.parametrize(
        "loss_func", ["smooth_mae", "mae", "mse", "rmse", "mape"]
    )  # loss function
    def test_consistency(self, prec, loss_func) -> None:
        """Construct -> forward -> serialize/deserialize -> forward -> compare.

        Also compare with dpmodel.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        nframes = 2
        task_dim = 5
        var_name = "foo"
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3
        natoms = 6

        loss0 = PropertyLoss(
            task_dim=task_dim,
            var_name=var_name,
            loss_func=loss_func,
            metric=["mae"],
            beta=1.0,
            out_bias=[0.1, 0.5, 1.2, -0.1, -10.0],
            out_std=[8.0, 10.0, 0.001, -0.2, -10.0],
            intensive=False,
        )

        model_pred, label = _make_data(
            rng, nframes, task_dim, var_name, dtype, self.device
        )

        # Forward
        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()

        # Serialize / deserialize round-trip
        loss1 = PropertyLoss.deserialize(loss0.serialize())
        l1, more1 = loss1(learning_rate, natoms, model_pred, label)

        np.testing.assert_allclose(
            l0.detach().cpu().numpy(),
            l1.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        for key in more0:
            np.testing.assert_allclose(
                more0[key].detach().cpu().numpy(),
                more1[key].detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"key={key}",
            )

        # Compare with dpmodel (numpy)
        dp_loss = PropertyLossDP.deserialize(loss0.serialize())
        model_pred_np = {k: v.detach().cpu().numpy() for k, v in model_pred.items()}
        label_np = {k: v.detach().cpu().numpy() for k, v in label.items()}
        l_dp, _more_dp = dp_loss(learning_rate, natoms, model_pred_np, label_np)

        # Use relative tolerance: extreme out_std values (e.g. 0.001) can produce
        # large loss values where torch/numpy accumulation order differs at machine epsilon.
        np.testing.assert_allclose(
            l0.detach().cpu().numpy(),
            np.array(l_dp),
            rtol=max(rtol, 1e-14 if prec == "float64" else 1e-5),
            atol=atol,
            err_msg="pt_expt vs dpmodel",
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_intensive(self, prec) -> None:
        """Test intensive property (no division by natoms)."""
        rng = np.random.default_rng(GLOBAL_SEED + 1)
        nframes = 2
        task_dim = 3
        var_name = "band_gap"
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3
        natoms = 6

        loss0 = PropertyLoss(
            task_dim=task_dim,
            var_name=var_name,
            loss_func="mse",
            metric=["mae", "rmse"],
            beta=1.0,
            out_bias=None,
            out_std=None,
            intensive=True,
        )

        model_pred, label = _make_data(
            rng, nframes, task_dim, var_name, dtype, self.device
        )

        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()
        assert "mae" in more0
        assert "rmse" in more0

        # Compare with dpmodel
        dp_loss = PropertyLossDP.deserialize(loss0.serialize())
        model_pred_np = {k: v.detach().cpu().numpy() for k, v in model_pred.items()}
        label_np = {k: v.detach().cpu().numpy() for k, v in label.items()}
        l_dp, _ = dp_loss(learning_rate, natoms, model_pred_np, label_np)

        np.testing.assert_allclose(
            l0.detach().cpu().numpy(),
            np.array(l_dp),
            rtol=rtol,
            atol=atol,
            err_msg="pt_expt vs dpmodel (intensive)",
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_no_out_bias_std(self, prec) -> None:
        """Test with out_bias and out_std as None (identity normalization)."""
        rng = np.random.default_rng(GLOBAL_SEED + 2)
        nframes = 2
        task_dim = 3
        var_name = "prop"
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3
        natoms = 6

        loss0 = PropertyLoss(
            task_dim=task_dim,
            var_name=var_name,
            loss_func="mae",
            out_bias=None,
            out_std=None,
            intensive=False,
        )

        model_pred, label = _make_data(
            rng, nframes, task_dim, var_name, dtype, self.device
        )

        l0, _ = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()

        # Compare with dpmodel
        dp_loss = PropertyLossDP.deserialize(loss0.serialize())
        model_pred_np = {k: v.detach().cpu().numpy() for k, v in model_pred.items()}
        label_np = {k: v.detach().cpu().numpy() for k, v in label.items()}
        l_dp, _ = dp_loss(learning_rate, natoms, model_pred_np, label_np)

        np.testing.assert_allclose(
            l0.detach().cpu().numpy(),
            np.array(l_dp),
            rtol=rtol,
            atol=atol,
            err_msg="pt_expt vs dpmodel (no bias/std)",
        )
