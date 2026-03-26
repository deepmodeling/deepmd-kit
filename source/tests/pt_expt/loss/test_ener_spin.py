# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the pt_expt EnergySpinLoss wrapper."""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.ener_spin import EnergySpinLoss as EnergySpinLossDP
from deepmd.pt_expt.loss.ener_spin import (
    EnergySpinLoss,
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
    n_magnetic: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build model prediction and label dicts as torch tensors."""
    # mask_mag: True for magnetic atoms, False otherwise
    mask_mag = torch.zeros((nframes, natoms, 1), dtype=torch.bool, device=device)
    mask_mag[:, :n_magnetic, :] = True
    model_pred = {
        "energy": torch.tensor(rng.random((nframes,)), dtype=dtype, device=device),
        "force": torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        ),
        "force_mag": torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        ),
        "mask_mag": mask_mag,
        "virial": torch.tensor(rng.random((nframes, 9)), dtype=dtype, device=device),
        "atom_energy": torch.tensor(
            rng.random((nframes, natoms)), dtype=dtype, device=device
        ),
    }
    label = {
        "energy": torch.tensor(rng.random((nframes,)), dtype=dtype, device=device),
        "force": torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        ),
        "force_mag": torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        ),
        "virial": torch.tensor(rng.random((nframes, 9)), dtype=dtype, device=device),
        "atom_ener": torch.tensor(
            rng.random((nframes, natoms)), dtype=dtype, device=device
        ),
        "find_energy": torch.tensor(1.0, dtype=dtype, device=device),
        "find_force": torch.tensor(1.0, dtype=dtype, device=device),
        "find_force_mag": torch.tensor(1.0, dtype=dtype, device=device),
        "find_virial": torch.tensor(1.0, dtype=dtype, device=device),
        "find_atom_ener": torch.tensor(1.0, dtype=dtype, device=device),
    }
    return model_pred, label


class TestEnergySpinLoss:
    def setup_method(self) -> None:
        self.device = env.DEVICE

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    @pytest.mark.parametrize("loss_func", ["mse", "mae"])  # loss function
    def test_consistency(self, prec, loss_func) -> None:
        """Construct -> forward -> serialize/deserialize -> forward -> compare.

        Also compare with dpmodel.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        nframes, natoms, n_magnetic = 2, 6, 4
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = EnergySpinLoss(
            starter_learning_rate=1e-3,
            start_pref_e=0.02,
            limit_pref_e=1.0,
            start_pref_fr=1000.0,
            limit_pref_fr=1.0,
            start_pref_fm=1000.0,
            limit_pref_fm=1.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            start_pref_ae=1.0,
            limit_pref_ae=1.0,
            loss_func=loss_func,
        ).to(self.device)

        model_pred, label = _make_data(
            rng, nframes, natoms, n_magnetic, dtype, self.device
        )

        # Forward
        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()
        assert "rmse" in more0

        # Serialize / deserialize round-trip
        loss1 = EnergySpinLoss.deserialize(loss0.serialize())
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
        dp_loss = EnergySpinLossDP.deserialize(loss0.serialize())
        model_pred_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in model_pred.items()
        }
        label_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in label.items()
        }
        l_dp, more_dp = dp_loss(learning_rate, natoms, model_pred_np, label_np)

        np.testing.assert_allclose(
            l0.detach().cpu().numpy(),
            np.array(l_dp),
            rtol=rtol,
            atol=atol,
            err_msg="pt_expt vs dpmodel",
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_partial_mask(self, prec) -> None:
        """Test with partial magnetic atoms (some atoms non-magnetic)."""
        rng = np.random.default_rng(GLOBAL_SEED + 1)
        nframes, natoms, n_magnetic = 2, 6, 2
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = EnergySpinLoss(
            starter_learning_rate=1e-3,
            start_pref_e=0.02,
            limit_pref_e=1.0,
            start_pref_fr=1000.0,
            limit_pref_fr=1.0,
            start_pref_fm=1000.0,
            limit_pref_fm=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
        ).to(self.device)

        model_pred, label = _make_data(
            rng, nframes, natoms, n_magnetic, dtype, self.device
        )

        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()
        assert "rmse_fm" in more0

        # Compare with dpmodel
        dp_loss = EnergySpinLossDP.deserialize(loss0.serialize())
        model_pred_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in model_pred.items()
        }
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
            err_msg="pt_expt vs dpmodel (partial mask)",
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_all_masked(self, prec) -> None:
        """Test with all atoms magnetic."""
        rng = np.random.default_rng(GLOBAL_SEED + 2)
        nframes, natoms, n_magnetic = 2, 6, 6
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = EnergySpinLoss(
            starter_learning_rate=1e-3,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=1000.0,
            limit_pref_fm=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
        ).to(self.device)

        model_pred, label = _make_data(
            rng, nframes, natoms, n_magnetic, dtype, self.device
        )

        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()

        # Compare with dpmodel
        dp_loss = EnergySpinLossDP.deserialize(loss0.serialize())
        model_pred_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in model_pred.items()
        }
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
            err_msg="pt_expt vs dpmodel (all masked)",
        )
