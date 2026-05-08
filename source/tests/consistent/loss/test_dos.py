# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.loss.dos import DOSLoss as DOSLossDP
from deepmd.utils.argcheck import (
    loss_dos,
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
    from deepmd.pt.loss.dos import DOSLoss as DOSLossPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    DOSLossPT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.loss.dos import DOSLoss as DOSLossPTExpt
else:
    DOSLossPTExpt = None
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict


@parameterized(
    (1.0, 0.0),  # pref_dos
    (1.0, 0.0),  # pref_ados
)
class TestDOS(CommonTest, LossTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (pref_dos, pref_ados) = self.param
        return {
            "start_pref_dos": pref_dos,
            "limit_pref_dos": pref_dos / 2 if pref_dos else 0.0,
            "start_pref_cdf": 0.0,
            "limit_pref_cdf": 0.0,
            "start_pref_ados": pref_ados,
            "limit_pref_ados": pref_ados / 2 if pref_ados else 0.0,
            "start_pref_acdf": 0.0,
            "limit_pref_acdf": 0.0,
        }

    skip_tf = True
    skip_pt = CommonTest.skip_pt
    skip_pt_expt = not INSTALLED_PT_EXPT
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT
    skip_pd = True

    dp_class = DOSLossDP
    pt_class = DOSLossPT
    pt_expt_class = DOSLossPTExpt
    jax_class = DOSLossDP
    array_api_strict_class = DOSLossDP
    args = loss_dos()

    def setUp(self) -> None:
        (pref_dos, pref_ados) = self.param
        if pref_dos == 0.0 and pref_ados == 0.0:
            self.skipTest("Both pref_dos and pref_ados are 0")
        CommonTest.setUp(self)
        self.learning_rate = 1e-3
        rng = np.random.default_rng(20250326)
        self.nframes = 2
        self.natoms = 6
        self.numb_dos = 4
        self.predict = {
            "dos": rng.random((self.nframes, self.numb_dos)),
            "atom_dos": rng.random((self.nframes, self.natoms, self.numb_dos)),
        }
        self.label = {
            "dos": rng.random((self.nframes, self.numb_dos)),
            "atom_dos": rng.random((self.nframes, self.natoms, self.numb_dos)),
            "find_dos": 1.0,
            "find_atom_dos": 1.0,
        }

    @property
    def additional_data(self) -> dict:
        return {
            "starter_learning_rate": 1e-3,
            "numb_dos": self.numb_dos,
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
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        predict = {kk: numpy_to_torch(vv) for kk, vv in self.predict.items()}
        label = {kk: numpy_to_torch(vv) for kk, vv in self.label.items()}
        loss, more_loss = pt_expt_obj(
            self.learning_rate,
            self.natoms,
            predict,
            label,
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
