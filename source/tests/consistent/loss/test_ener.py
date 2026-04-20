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
    INSTALLED_PT_EXPT,
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
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.loss.ener import EnergyLoss as EnerLossPTExpt
else:
    EnerLossPTExpt = None
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict


@parameterized(
    (False, True),  # huber
    (False, True),  # enable_atom_ener_coeff
    ("mse", "mae"),  # loss_func
    (False, True),  # f_use_norm
    (False, True),  # mae (dp test extra MAE metrics)
    (False, True),  # intensive
)
class TestEner(CommonTest, LossTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            use_huber,
            enable_atom_ener_coeff,
            loss_func,
            f_use_norm,
            _mae,
            intensive,
        ) = self.param
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
            "enable_atom_ener_coeff": enable_atom_ener_coeff,
            "loss_func": loss_func,
            "f_use_norm": f_use_norm,
            "intensive": intensive,
        }

    @property
    def skip_tf(self) -> bool:
        (
            _use_huber,
            _enable_atom_ener_coeff,
            loss_func,
            f_use_norm,
            _mae,
            _intensive,
        ) = self.param
        # Skip TF for MAE loss tests (not implemented in TF backend)
        return CommonTest.skip_tf or loss_func == "mae" or f_use_norm

    @property
    def skip_pd(self) -> bool:
        (
            _use_huber,
            _enable_atom_ener_coeff,
            loss_func,
            f_use_norm,
            _mae,
            _intensive,
        ) = self.param
        # Skip Paddle for MAE loss tests (not implemented in Paddle backend)
        return not INSTALLED_PD or loss_func == "mae" or f_use_norm

    skip_pt = CommonTest.skip_pt
    skip_pt_expt = not INSTALLED_PT_EXPT
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    tf_class = EnerLossTF
    dp_class = EnerLossDP
    pt_class = EnerLossPT
    pt_expt_class = EnerLossPTExpt
    jax_class = EnerLossDP
    pd_class = EnerLossPD
    array_api_strict_class = EnerLossDP
    args = loss_ener()

    def setUp(self) -> None:
        (
            use_huber,
            _enable_atom_ener_coeff,
            loss_func,
            f_use_norm,
            mae,
            _intensive,
        ) = self.param
        # Skip invalid combinations
        if f_use_norm and not (use_huber or loss_func == "mae"):
            self.skipTest("f_use_norm requires either use_huber or loss_func='mae'")
        if use_huber and loss_func == "mae":
            self.skipTest("Cannot use both huber and mae loss_func at the same time")
        if loss_func == "mae" and mae:
            self.skipTest("mae=True with loss_func='mae' is redundant")
        CommonTest.setUp(self)
        self.mae = mae
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
            "energy": self.predict["energy"],
            "force": self.predict["force"],
            "virial": self.predict["virial"],
            "atom_energy": self.predict["atom_ener"],
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
            "atom_ener_coeff": rng.random((self.nframes, self.natoms)),
            "atom_pref": np.ones((self.nframes, self.natoms, 3)),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 1.0,
            "find_atom_ener_coeff": 1.0,
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

        loss, _more_loss = obj.build(
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
            mae=self.mae,
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
            mae=self.mae,
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        predict = {
            kk: numpy_to_torch(vv) for kk, vv in self.predict_dpmodel_style.items()
        }
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
        predict = {kk: jnp.asarray(vv) for kk, vv in self.predict_dpmodel_style.items()}
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
            mae=self.mae,
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
        """Relative tolerance for comparing the return value."""
        return 1e-10

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        return 1e-10


class TestEnerGF(CommonTest, LossTest, unittest.TestCase):
    """Test energy loss with generalized force (numb_generalized_coord > 0).

    This exercises the code path that previously had a natoms[0] bug.
    """

    @property
    def data(self) -> dict:
        return {
            "start_pref_e": 0.02,
            "limit_pref_e": 1.0,
            "start_pref_f": 1000.0,
            "limit_pref_f": 1.0,
            "start_pref_v": 1.0,
            "limit_pref_v": 1.0,
            "start_pref_ae": 1.0,
            "limit_pref_ae": 1.0,
            "start_pref_pf": 1.0,
            "limit_pref_pf": 1.0,
            "start_pref_gf": 1.0,
            "limit_pref_gf": 1.0,
            "numb_generalized_coord": 2,
        }

    skip_tf = CommonTest.skip_tf
    skip_pt = CommonTest.skip_pt
    skip_pt_expt = not INSTALLED_PT_EXPT
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT
    skip_pd = not INSTALLED_PD

    tf_class = EnerLossTF
    dp_class = EnerLossDP
    pt_class = EnerLossPT
    pt_expt_class = EnerLossPTExpt
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
        numb_generalized_coord = 2
        self.predict = {
            "energy": rng.random((self.nframes,)),
            "force": rng.random((self.nframes, self.natoms, 3)),
            "virial": rng.random((self.nframes, 9)),
            "atom_ener": rng.random((self.nframes, self.natoms)),
        }
        self.predict_dpmodel_style = {
            "energy": self.predict["energy"],
            "force": self.predict["force"],
            "virial": self.predict["virial"],
            "atom_energy": self.predict["atom_ener"],
        }
        self.label = {
            "energy": rng.random((self.nframes,)),
            "force": rng.random((self.nframes, self.natoms, 3)),
            "virial": rng.random((self.nframes, 9)),
            "atom_ener": rng.random((self.nframes, self.natoms)),
            "atom_pref": np.ones((self.nframes, self.natoms, 3)),
            "drdq": rng.random(
                (self.nframes, self.natoms * 3 * numb_generalized_coord)
            ),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 1.0,
            "find_atom_pref": 1.0,
            "find_drdq": 1.0,
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

        loss, _more_loss = obj.build(
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
            mae=True,
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
            mae=True,
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        predict = {
            kk: numpy_to_torch(vv) for kk, vv in self.predict_dpmodel_style.items()
        }
        label = {kk: numpy_to_torch(vv) for kk, vv in self.label.items()}
        loss, more_loss = pt_expt_obj(
            self.learning_rate,
            self.natoms,
            predict,
            label,
            mae=True,
        )
        loss = torch_to_numpy(loss)
        more_loss = {kk: torch_to_numpy(vv) for kk, vv in more_loss.items()}
        return loss, more_loss

    def eval_jax(self, jax_obj: Any) -> Any:
        predict = {kk: jnp.asarray(vv) for kk, vv in self.predict_dpmodel_style.items()}
        label = {kk: jnp.asarray(vv) for kk, vv in self.label.items()}

        loss, more_loss = jax_obj(
            self.learning_rate,
            self.natoms,
            predict,
            label,
            mae=True,
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
            mae=True,
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
            mae=True,
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


class TestIntensiveNatomsScaling(unittest.TestCase):
    """Regression test for natoms-scaling behavior with intensive normalization.

    This test verifies that MSE energy/virial loss contributions scale with 1/N² when
    intensive=True, ensuring the loss is independent of system size. This guards against
    future refactors accidentally reverting to 1/N scaling.
    """

    def test_intensive_total_loss_scaling(self) -> None:
        """Test that total loss scales correctly with 1/N² for intensive=True.

        This test uses controlled energy/virial residuals to verify that the
        total loss contribution scales with 1/N² (intensive) vs 1/N (legacy).
        We use identical per-atom residuals across different system sizes to
        ensure the raw MSE is the same, then verify the total loss scales as
        expected based on the normalization factor.
        """
        if not INSTALLED_PT:
            self.skipTest("PyTorch not installed")

        nframes = 1

        # Test with two different system sizes
        natoms_small = 4
        natoms_large = 8  # 2x the small system

        # Use fixed energy residual so MSE is predictable
        # Energy residual = 1.0, so l2_ener_loss = 1.0
        fixed_energy_diff = 1.0

        def create_data_with_fixed_residual(natoms: int, energy_diff: float):
            """Create predict/label with a fixed energy difference."""
            predict = {
                "energy": numpy_to_torch(np.array([1.0])),
                "force": numpy_to_torch(np.zeros((nframes, natoms, 3))),
                "virial": numpy_to_torch(np.array([[1.0] * 9])),  # Virial residual = 1
                "atom_energy": numpy_to_torch(np.ones((nframes, natoms)) / natoms),
            }
            label = {
                "energy": numpy_to_torch(np.array([1.0 + energy_diff])),
                "force": numpy_to_torch(np.zeros((nframes, natoms, 3))),
                "virial": numpy_to_torch(np.array([[2.0] * 9])),  # Virial residual = 1
                "atom_ener": numpy_to_torch(
                    np.ones((nframes, natoms)) * (1.0 + energy_diff) / natoms
                ),
                "find_energy": 1.0,
                "find_force": 0.0,  # Disable force to focus on energy/virial
                "find_virial": 1.0,
                "find_atom_ener": 0.0,
            }
            return predict, label

        # Create loss functions
        loss_intensive = EnerLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive=True,
        )
        loss_legacy = EnerLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive=False,
        )

        # Compute losses for small system
        predict_small, label_small = create_data_with_fixed_residual(
            natoms_small, fixed_energy_diff
        )
        _, loss_intensive_small, _ = loss_intensive(
            {},
            lambda p=predict_small: p,
            label_small,
            natoms_small,
            1e-3,
        )
        _, loss_legacy_small, _ = loss_legacy(
            {},
            lambda p=predict_small: p,
            label_small,
            natoms_small,
            1e-3,
        )

        # Compute losses for large system
        predict_large, label_large = create_data_with_fixed_residual(
            natoms_large, fixed_energy_diff
        )
        _, loss_intensive_large, _ = loss_intensive(
            {},
            lambda p=predict_large: p,
            label_large,
            natoms_large,
            1e-3,
        )
        _, loss_legacy_large, _ = loss_legacy(
            {},
            lambda p=predict_large: p,
            label_large,
            natoms_large,
            1e-3,
        )

        loss_int_small = float(torch_to_numpy(loss_intensive_small))
        loss_int_large = float(torch_to_numpy(loss_intensive_large))
        loss_leg_small = float(torch_to_numpy(loss_legacy_small))
        loss_leg_large = float(torch_to_numpy(loss_legacy_large))

        # With same residuals but different natoms:
        # - intensive (1/N²): loss should scale as (N_small/N_large)² = (4/8)² = 0.25
        # - legacy (1/N): loss should scale as (N_small/N_large) = 4/8 = 0.5

        # Verify intensive scaling: loss_large / loss_small should be ~0.25
        natoms_ratio = natoms_small / natoms_large  # 0.5
        expected_intensive_ratio = natoms_ratio**2  # 0.25
        expected_legacy_ratio = natoms_ratio  # 0.5

        actual_intensive_ratio = loss_int_large / loss_int_small
        actual_legacy_ratio = loss_leg_large / loss_leg_small

        self.assertAlmostEqual(
            actual_intensive_ratio,
            expected_intensive_ratio,
            places=5,
            msg=f"Intensive loss scaling: expected {expected_intensive_ratio:.4f}, "
            f"got {actual_intensive_ratio:.4f}",
        )
        self.assertAlmostEqual(
            actual_legacy_ratio,
            expected_legacy_ratio,
            places=5,
            msg=f"Legacy loss scaling: expected {expected_legacy_ratio:.4f}, "
            f"got {actual_legacy_ratio:.4f}",
        )

    def test_intensive_vs_legacy_scaling_difference(self) -> None:
        """Test that intensive=True produces different loss than intensive=False for energy/virial."""
        if not INSTALLED_PT:
            self.skipTest("PyTorch not installed")

        rng = np.random.default_rng(20250419)
        nframes = 1
        natoms = 8

        predict = {
            "energy": numpy_to_torch(rng.random((nframes,))),
            "force": numpy_to_torch(rng.random((nframes, natoms, 3))),
            "virial": numpy_to_torch(rng.random((nframes, 9))),
            "atom_energy": numpy_to_torch(rng.random((nframes, natoms))),
        }
        label = {
            "energy": numpy_to_torch(rng.random((nframes,))),
            "force": numpy_to_torch(rng.random((nframes, natoms, 3))),
            "virial": numpy_to_torch(rng.random((nframes, 9))),
            "atom_ener": numpy_to_torch(rng.random((nframes, natoms))),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 0.0,
        }

        # Create loss functions with intensive=True and intensive=False
        loss_intensive = EnerLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive=True,
        )
        loss_legacy = EnerLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive=False,
        )

        _, loss_val_intensive, _ = loss_intensive(
            {},
            lambda: predict,
            label,
            natoms,
            1e-3,
        )
        _, loss_val_legacy, _ = loss_legacy(
            {},
            lambda: predict,
            label,
            natoms,
            1e-3,
        )

        loss_intensive_val = float(torch_to_numpy(loss_val_intensive))
        loss_legacy_val = float(torch_to_numpy(loss_val_legacy))

        # The losses should be different when intensive differs
        # (unless by chance the values are the same, which is unlikely)
        # The intensive version should have an extra 1/N factor
        expected_ratio = 1.0 / natoms
        actual_ratio = loss_intensive_val / loss_legacy_val

        # Allow some tolerance due to floating point
        self.assertAlmostEqual(
            actual_ratio,
            expected_ratio,
            places=5,
            msg=f"Expected intensive/legacy ratio ~{expected_ratio:.6f}, got {actual_ratio:.6f}",
        )
