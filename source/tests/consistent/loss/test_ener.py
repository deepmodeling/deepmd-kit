# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.loss.ener import EnergyLoss as EnerLossDP
from deepmd.utils.argcheck import (
    loss_ener,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import (
    LossTest,
)

if INSTALLED_TF:
    from deepmd.tf.env import (
        GLOBAL_TF_FLOAT_PRECISION,
        tf,
    )
    from deepmd.tf.loss.ener import EnerStdLoss as EnerLossTF
else:
    EnerLossTF = None
if INSTALLED_PT:
    from deepmd.pt.loss.ener import EnergyStdLoss as EnerLossPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    EnerLossPT = None
if INSTALLED_PD:
    import paddle

    from deepmd.pd.loss.ener import EnergyStdLoss as EnerLossPD
    from deepmd.pd.utils.env import DEVICE as PD_DEVICE
else:
    EnerLossPD = None
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict


@parameterized(
    (False, True),  # use_huber
)
class TestEner(CommonTest, LossTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (use_huber,) = self.param
        return {
            "start_pref_e": 0.02,
            "limit_pref_e": 1.0,
            "start_pref_f": 1000.0,
            "limit_pref_f": 1.0,
            "start_pref_v": 1.0,
            "limit_pref_v": 1.0,
            "start_pref_ae": 1.0,
            "limit_pref_ae": 1.0,
            "start_pref_pf": 1.0 if not use_huber else 0.0,
            "limit_pref_pf": 1.0 if not use_huber else 0.0,
            "use_huber": use_huber,
        }

    skip_tf = CommonTest.skip_tf
    skip_pt = CommonTest.skip_pt
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT
    skip_pd = not INSTALLED_PD

    tf_class = EnerLossTF
    dp_class = EnerLossDP
    pt_class = EnerLossPT
    jax_class = EnerLossDP
    pd_class = EnerLossPD
    array_api_strict_class = EnerLossDP
    args = loss_ener()

    def setUp(self) -> None:
        CommonTest.setUp(self)
        self.learning_rate = 1e-3
        rng = np.random.default_rng(20250105)
        self.nframes = 2
        self.natoms = 6
        self.predict = {
            "energy": rng.random((self.nframes,)),
            "force": rng.random((self.nframes, self.natoms, 3)),
            "virial": rng.random((self.nframes, 9)),
            "atom_ener": rng.random(
                (
                    self.nframes,
                    self.natoms,
                )
            ),
        }
        self.predict_dpmodel_style = {
            "energy_derv_c_redu": self.predict["virial"],
            "energy_derv_r": self.predict["force"],
            "energy_redu": self.predict["energy"],
            "energy": self.predict["atom_ener"],
        }
        self.label = {
            "energy": rng.random((self.nframes,)),
            "force": rng.random((self.nframes, self.natoms, 3)),
            "virial": rng.random((self.nframes, 9)),
            "atom_ener": rng.random(
                (
                    self.nframes,
                    self.natoms,
                )
            ),
            "atom_pref": np.ones((self.nframes, self.natoms, 3)),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 1.0,
            "find_atom_pref": 1.0,
        }

    @property
    def additional_data(self) -> dict:
        return {
            "starter_learning_rate": 1e-3,
        }

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        predict = {
            kk: tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION, vv.shape, name="i_predict_" + kk
            )
            for kk, vv in self.predict.items()
        }
        label = {
            kk: tf.placeholder(
                GLOBAL_TF_FLOAT_PRECISION, vv.shape, name="i_label_" + kk
            )
            if isinstance(vv, np.ndarray)
            else vv
            for kk, vv in self.label.items()
        }

        loss, more_loss = obj.build(
            self.learning_rate,
            [self.natoms],
            predict,
            label,
            suffix=suffix,
        )
        return [loss], {
            **{
                vv: self.predict[kk]
                for kk, vv in predict.items()
                if isinstance(vv, tf.Tensor)
            },
            **{
                vv: self.label[kk]
                for kk, vv in label.items()
                if isinstance(vv, tf.Tensor)
            },
        }

    def eval_pt(self, pt_obj: Any) -> Any:
        predict = {kk: numpy_to_torch(vv) for kk, vv in self.predict.items()}
        label = {kk: numpy_to_torch(vv) for kk, vv in self.label.items()}
        predict["atom_energy"] = predict.pop("atom_ener")
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
            self.predict_dpmodel_style,
            self.label,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        predict = {kk: jnp.asarray(vv) for kk, vv in self.predict_dpmodel_style.items()}
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
        predict = {
            kk: array_api_strict.asarray(vv)
            for kk, vv in self.predict_dpmodel_style.items()
        }
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

    def eval_pd(self, pd_obj: Any) -> Any:
        predict = {
            kk: paddle.to_tensor(vv).to(PD_DEVICE) for kk, vv in self.predict.items()
        }
        label = {
            kk: paddle.to_tensor(vv).to(PD_DEVICE) for kk, vv in self.label.items()
        }
        predict["atom_energy"] = predict.pop("atom_ener")
        _, loss, more_loss = pd_obj(
            {},
            lambda: predict,
            label,
            self.natoms,
            self.learning_rate,
        )
        loss = to_numpy_array(loss)
        more_loss = {kk: to_numpy_array(vv) for kk, vv in more_loss.items()}
        return loss, more_loss

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        return 1e-10

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        return 1e-10
