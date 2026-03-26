# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the pt_expt DOSLoss wrapper."""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.dos import DOSLoss as DOSLossDP
from deepmd.pt_expt.loss.dos import (
    DOSLoss,
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
    natoms: int,
    numb_dos: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build model prediction and label dicts as torch tensors."""
    model_pred = {
        "dos": torch.tensor(
            rng.random((nframes, numb_dos)), dtype=dtype, device=device
        ),
        "atom_dos": torch.tensor(
            rng.random((nframes, natoms, numb_dos)), dtype=dtype, device=device
        ),
    }
    label = {
        "dos": torch.tensor(
            rng.random((nframes, numb_dos)), dtype=dtype, device=device
        ),
        "atom_dos": torch.tensor(
            rng.random((nframes, natoms, numb_dos)), dtype=dtype, device=device
        ),
        "find_dos": torch.tensor(1.0, dtype=dtype, device=device),
        "find_atom_dos": torch.tensor(1.0, dtype=dtype, device=device),
    }
    return model_pred, label


class TestDOSLoss:
    def setup_method(self) -> None:
        self.device = env.DEVICE

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    @pytest.mark.parametrize(
        "has_dos,has_ados",
        [(True, True), (True, False), (False, True)],
    )  # which loss terms are active
    def test_consistency(self, prec, has_dos, has_ados) -> None:
        """Construct -> forward -> serialize/deserialize -> forward -> compare.

        Also compare with dpmodel.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        nframes, natoms, numb_dos = 2, 6, 4
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = DOSLoss(
            starter_learning_rate=1e-3,
            numb_dos=numb_dos,
            start_pref_dos=1.0 if has_dos else 0.0,
            limit_pref_dos=0.5 if has_dos else 0.0,
            start_pref_cdf=0.0,
            limit_pref_cdf=0.0,
            start_pref_ados=1.0 if has_ados else 0.0,
            limit_pref_ados=0.5 if has_ados else 0.0,
            start_pref_acdf=0.0,
            limit_pref_acdf=0.0,
        ).to(self.device)

        model_pred, label = _make_data(
            rng, nframes, natoms, numb_dos, dtype, self.device
        )

        # Forward
        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()
        assert "rmse" in more0

        # Serialize / deserialize round-trip
        loss1 = DOSLoss.deserialize(loss0.serialize())
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
        dp_loss = DOSLossDP.deserialize(loss0.serialize())
        model_pred_np = {k: v.detach().cpu().numpy() for k, v in model_pred.items()}
        label_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in label.items()
        }
        l_dp, _more_dp = dp_loss(learning_rate, natoms, model_pred_np, label_np)

        np.testing.assert_allclose(
            l0.detach().cpu().numpy(),
            np.array(l_dp),
            rtol=rtol,
            atol=atol,
            err_msg="pt_expt vs dpmodel",
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_cdf_terms(self, prec) -> None:
        """Test with CDF loss terms enabled."""
        rng = np.random.default_rng(GLOBAL_SEED + 1)
        nframes, natoms, numb_dos = 2, 6, 4
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = DOSLoss(
            starter_learning_rate=1e-3,
            numb_dos=numb_dos,
            start_pref_dos=0.0,
            limit_pref_dos=0.0,
            start_pref_cdf=1.0,
            limit_pref_cdf=0.5,
            start_pref_ados=0.0,
            limit_pref_ados=0.0,
            start_pref_acdf=1.0,
            limit_pref_acdf=0.5,
        ).to(self.device)

        model_pred, label = _make_data(
            rng, nframes, natoms, numb_dos, dtype, self.device
        )

        l0, _more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()

        # Compare with dpmodel
        dp_loss = DOSLossDP.deserialize(loss0.serialize())
        model_pred_np = {k: v.detach().cpu().numpy() for k, v in model_pred.items()}
        label_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in label.items()
        }
        l_dp, _ = dp_loss(learning_rate, natoms, model_pred_np, label_np)

        np.testing.assert_allclose(
            l0.detach().cpu().numpy(),
            np.array(l_dp),
            rtol=rtol,
            atol=atol,
            err_msg="pt_expt vs dpmodel (cdf terms)",
        )
