# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the pt_expt TensorLoss wrapper."""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.loss.tensor import TensorLoss as TensorLossDP
from deepmd.pt_expt.loss.tensor import (
    TensorLoss,
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
    tensor_name: str,
    label_name: str,
    tensor_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build model prediction and label dicts as torch tensors."""
    model_pred = {
        tensor_name: torch.tensor(
            rng.random((nframes, natoms, tensor_size)), dtype=dtype, device=device
        ),
        "global_" + tensor_name: torch.tensor(
            rng.random((nframes, tensor_size)), dtype=dtype, device=device
        ),
    }
    label = {
        "atom_" + label_name: torch.tensor(
            rng.random((nframes, natoms, tensor_size)), dtype=dtype, device=device
        ),
        label_name: torch.tensor(
            rng.random((nframes, tensor_size)), dtype=dtype, device=device
        ),
        "find_atom_" + label_name: torch.tensor(1.0, dtype=dtype, device=device),
        "find_" + label_name: torch.tensor(1.0, dtype=dtype, device=device),
    }
    return model_pred, label


class TestTensorLoss:
    def setup_method(self) -> None:
        self.device = env.DEVICE

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    @pytest.mark.parametrize(
        "has_local,has_global",
        [(True, True), (True, False), (False, True)],
    )  # which loss terms are active
    def test_consistency(self, prec, has_local, has_global) -> None:
        """Construct -> forward -> serialize/deserialize -> forward -> compare.

        Also compare with dpmodel.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        nframes, natoms = 2, 6
        tensor_name = "test_tensor"
        label_name = "test_tensor"
        tensor_size = 3
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = TensorLoss(
            tensor_name=tensor_name,
            tensor_size=tensor_size,
            label_name=label_name,
            pref_atomic=1.0 if has_local else 0.0,
            pref=1.0 if has_global else 0.0,
        )

        model_pred, label = _make_data(
            rng,
            nframes,
            natoms,
            tensor_name,
            label_name,
            tensor_size,
            dtype,
            self.device,
        )

        # Forward
        l0, more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()
        assert "rmse" in more0

        # Serialize / deserialize round-trip
        loss1 = TensorLoss.deserialize(loss0.serialize())
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
        dp_loss = TensorLossDP.deserialize(loss0.serialize())
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
    def test_with_atomic_weight(self, prec) -> None:
        """Test with atomic weight enabled."""
        rng = np.random.default_rng(GLOBAL_SEED + 1)
        nframes, natoms = 2, 6
        tensor_name = "dipole"
        label_name = "dipole"
        tensor_size = 3
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        learning_rate = 1e-3

        loss0 = TensorLoss(
            tensor_name=tensor_name,
            tensor_size=tensor_size,
            label_name=label_name,
            pref_atomic=1.0,
            pref=1.0,
            enable_atomic_weight=True,
        )

        model_pred, label = _make_data(
            rng,
            nframes,
            natoms,
            tensor_name,
            label_name,
            tensor_size,
            dtype,
            self.device,
        )
        label["atom_weight"] = torch.tensor(
            rng.random((nframes, natoms)), dtype=dtype, device=self.device
        )

        l0, _more0 = loss0(learning_rate, natoms, model_pred, label)
        assert l0.shape == ()

        # Compare with dpmodel
        dp_loss = TensorLossDP.deserialize(loss0.serialize())
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
            err_msg="pt_expt vs dpmodel (atomic weight)",
        )
