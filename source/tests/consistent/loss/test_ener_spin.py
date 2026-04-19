# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.loss.ener_spin import EnergySpinLoss as EnerSpinLossDP
from deepmd.utils.argcheck import (
    loss_ener_spin,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    CommonTest,
    parameterized,
)
from .common import (
    LossTest,
)

if INSTALLED_PT:
    from deepmd.pt.loss.ener_spin import EnergySpinLoss as EnerSpinLossPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    EnerSpinLossPT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.loss.ener_spin import EnergySpinLoss as EnerSpinLossPTExpt
else:
    EnerSpinLossPTExpt = None
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict


@parameterized(
    ("mse", "mae"),  # loss_func
    (False, True),  # mae (dp test extra MAE metrics)
    (False, True),  # intensive
)
class TestEnerSpin(CommonTest, LossTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (loss_func, _mae, intensive) = self.param
        return {
            "start_pref_e": 0.02,
            "limit_pref_e": 1.0,
            "start_pref_fr": 1000.0,
            "limit_pref_fr": 1.0,
            "start_pref_fm": 1000.0,
            "limit_pref_fm": 1.0,
            "start_pref_v": 1.0,
            "limit_pref_v": 1.0,
            "start_pref_ae": 1.0,
            "limit_pref_ae": 1.0,
            "loss_func": loss_func,
            "intensive": intensive,
        }

    skip_tf = True
    skip_pt = CommonTest.skip_pt
    skip_pt_expt = not INSTALLED_PT_EXPT
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT
    skip_pd = True

    dp_class = EnerSpinLossDP
    pt_class = EnerSpinLossPT
    pt_expt_class = EnerSpinLossPTExpt
    jax_class = EnerSpinLossDP
    array_api_strict_class = EnerSpinLossDP
    args = loss_ener_spin()

    def setUp(self) -> None:
        (loss_func, mae, _intensive) = self.param
        if loss_func == "mae" and mae:
            self.skipTest("mae=True with loss_func='mae' is redundant")
        CommonTest.setUp(self)
        self.mae = mae
        self.learning_rate = 1e-3
        rng = np.random.default_rng(20250326)
        self.nframes = 2
        self.natoms = 6
        n_magnetic = 4
        mask_mag = np.zeros((self.nframes, self.natoms, 1), dtype=bool)
        mask_mag[:, :n_magnetic, :] = True
        self.predict = {
            "energy": rng.random((self.nframes,)),
            "force": rng.random((self.nframes, self.natoms, 3)),
            "force_mag": rng.random((self.nframes, self.natoms, 3)),
            "mask_mag": mask_mag,
            "virial": rng.random((self.nframes, 9)),
            "atom_energy": rng.random((self.nframes, self.natoms)),
        }
        self.label = {
            "energy": rng.random((self.nframes,)),
            "force": rng.random((self.nframes, self.natoms, 3)),
            "force_mag": rng.random((self.nframes, self.natoms, 3)),
            "virial": rng.random((self.nframes, 9)),
            "atom_ener": rng.random((self.nframes, self.natoms)),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_force_mag": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 1.0,
        }

    @property
    def additional_data(self) -> dict:
        return {
            "starter_learning_rate": 1e-3,
        }

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError

    def eval_pt(self, pt_obj: Any) -> Any:
        predict = {kk: numpy_to_torch(vv) for kk, vv in self.predict.items()}
        label = {kk: numpy_to_torch(vv) for kk, vv in self.label.items()}
        _, loss, more_loss = pt_obj(
            {},
            lambda: predict,
            label,
            self.natoms,
            self.learning_rate,
            mae=self.mae,
        )
        loss = torch_to_numpy(loss)
        more_loss = {kk: torch_to_numpy(vv) for kk, vv in more_loss.items()}
        return loss, more_loss

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj(
            self.learning_rate,
            self.natoms,
            self.predict,
            self.label,
            mae=self.mae,
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        predict = {kk: numpy_to_torch(vv) for kk, vv in self.predict.items()}
        label = {kk: numpy_to_torch(vv) for kk, vv in self.label.items()}
        loss, more_loss = pt_expt_obj(
            self.learning_rate,
            self.natoms,
            predict,
            label,
            mae=self.mae,
        )
        loss = torch_to_numpy(loss)
        more_loss = {kk: torch_to_numpy(vv) for kk, vv in more_loss.items()}
        return loss, more_loss

    def eval_jax(self, jax_obj: Any) -> Any:
        predict = {kk: jnp.asarray(vv) for kk, vv in self.predict.items()}
        label = {kk: jnp.asarray(vv) for kk, vv in self.label.items()}
        loss, more_loss = jax_obj(
            self.learning_rate,
            self.natoms,
            predict,
            label,
            mae=self.mae,
        )
        loss = to_numpy_array(loss)
        more_loss = {kk: to_numpy_array(vv) for kk, vv in more_loss.items()}
        return loss, more_loss

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        predict = {kk: array_api_strict.asarray(vv) for kk, vv in self.predict.items()}
        label = {kk: array_api_strict.asarray(vv) for kk, vv in self.label.items()}
        loss, more_loss = array_api_strict_obj(
            self.learning_rate,
            self.natoms,
            predict,
            label,
            mae=self.mae,
        )
        loss = to_numpy_array(loss)
        more_loss = {kk: to_numpy_array(vv) for kk, vv in more_loss.items()}
        return loss, more_loss

    def extract_ret(self, ret: Any, backend) -> dict[str, np.ndarray]:
        loss = ret[0]
        result = {"loss": np.atleast_1d(np.asarray(loss, dtype=np.float64))}
        if len(ret) > 1:
            more_loss = ret[1]
            for k in sorted(more_loss):
                if k.startswith("rmse_") or k.startswith("mae_"):
                    result[k] = np.atleast_1d(
                        np.asarray(more_loss[k], dtype=np.float64)
                    )
        return result

    @property
    def rtol(self) -> float:
        return 1e-10

    @property
    def atol(self) -> float:
        return 1e-10
