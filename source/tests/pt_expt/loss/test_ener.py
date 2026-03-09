# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the pt_expt EnergyLoss wrapper.

Three test types:
- test_consistency — construct -> forward -> serialize/deserialize -> forward -> compare;
  also compare with dpmodel
- test_consistency_with_find_flags — same but with find_* flags as torch tensors
  (mimicking real training where get_data converts them)
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.ener import EnergyLoss as EnergyLossDP
from deepmd.pt_expt.loss.ener import (
    EnergyLoss,
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
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build model prediction and label dicts as torch tensors."""
    model_pred = {
        "energy": torch.tensor(rng.random((nframes,)), dtype=dtype, device=device),
        "force": torch.tensor(
            rng.random((nframes, natoms, 3)), dtype=dtype, device=device
        ),
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
        "virial": torch.tensor(rng.random((nframes, 9)), dtype=dtype, device=device),
        "atom_ener": torch.tensor(
            rng.random((nframes, natoms)), dtype=dtype, device=device
        ),
        "atom_pref": torch.ones((nframes, natoms, 3), dtype=dtype, device=device),
        "find_energy": torch.tensor(1.0, dtype=dtype, device=device),
        "find_force": torch.tensor(1.0, dtype=dtype, device=device),
        "find_virial": torch.tensor(1.0, dtype=dtype, device=device),
        "find_atom_ener": torch.tensor(1.0, dtype=dtype, device=device),
        "find_atom_pref": torch.tensor(1.0, dtype=dtype, device=device),
    }
    return model_pred, label


class TestEnergyLoss:
    def setup_method(self) -> None:
        self.device = env.DEVICE

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    @pytest.mark.parametrize("use_huber", [False, True])  # huber loss
    def test_consistency(self, prec, use_huber) -> None:
        """Construct -> forward -> serialize/deserialize -> forward -> compare.

        Also compare with dpmodel.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        nframes, natoms = 2, 6
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = EnergyLoss(
            starter_learning_rate=1e-3,
            start_pref_e=0.02,
            limit_pref_e=1.0,
            start_pref_f=1000.0,
            limit_pref_f=1.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            start_pref_ae=1.0,
            limit_pref_ae=1.0,
            start_pref_pf=0.0 if use_huber else 1.0,
            limit_pref_pf=0.0 if use_huber else 1.0,
            use_huber=use_huber,
        ).to(self.device)

        model_pred, label = _make_data(rng, nframes, natoms, dtype, self.device)

        # Forward
        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()
        assert "rmse" in more0

        # Serialize / deserialize round-trip
        loss1 = EnergyLoss.deserialize(loss0.serialize())
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
        dp_loss = EnergyLossDP.deserialize(loss0.serialize())
        model_pred_np = {k: v.detach().cpu().numpy() for k, v in model_pred.items()}
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
