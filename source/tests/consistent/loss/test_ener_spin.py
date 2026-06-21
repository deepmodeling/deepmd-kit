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
    (False, True),  # intensive_ener_virial
)
class TestEnerSpin(CommonTest, LossTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (loss_func, _mae, intensive_ener_virial) = self.param
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
            "intensive_ener_virial": intensive_ener_virial,
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
        (loss_func, mae, intensive_ener_virial) = self.param
        if loss_func == "mae" and mae:
            self.skipTest("mae=True with loss_func='mae' is redundant")
        CommonTest.setUp(self)
        self.mae = mae
        self.intensive_ener_virial = intensive_ener_virial
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


class TestEnerSpinIntensiveScaling(unittest.TestCase):
    """Regression test for natoms-scaling behavior with intensive normalization.

    This test verifies that MSE energy/virial loss contributions scale with 1/N² when
    intensive_ener_virial=True, ensuring the loss is independent of system size. This guards against
    future refactors accidentally reverting to 1/N scaling.
    """

    def test_intensive_total_loss_scaling(self) -> None:
        """Test that total loss scales correctly with 1/N² for intensive_ener_virial=True.

        This test uses controlled energy/virial residuals to verify that the
        total loss contribution scales with 1/N² (intensive) vs 1/N (legacy).
        """
        if not INSTALLED_PT:
            self.skipTest("PyTorch not installed")

        nframes = 1

        # Test with two different system sizes
        natoms_small = 4
        natoms_large = 8  # 2x the small system
        # For spin systems, we have real atoms and virtual (magnetic) atoms
        n_magnetic = 2  # Half of atoms have magnetic spins

        # Use fixed energy residual so MSE is predictable
        fixed_energy_diff = 1.0

        def create_data_with_fixed_residual(
            natoms: int, n_mag: int, energy_diff: float
        ):
            """Create predict/label with a fixed energy difference."""
            mask_mag = np.zeros((nframes, natoms, 1), dtype=bool)
            mask_mag[:, :n_mag, :] = True

            predict = {
                "energy": numpy_to_torch(np.array([1.0])),
                "force": numpy_to_torch(np.zeros((nframes, natoms, 3))),
                "force_mag": numpy_to_torch(np.zeros((nframes, natoms, 3))),
                "mask_mag": mask_mag,
                "virial": numpy_to_torch(np.array([[1.0] * 9])),
                "atom_energy": numpy_to_torch(np.ones((nframes, natoms)) / natoms),
            }
            label = {
                "energy": numpy_to_torch(np.array([1.0 + energy_diff])),
                "force": numpy_to_torch(np.zeros((nframes, natoms, 3))),
                "force_mag": numpy_to_torch(np.zeros((nframes, natoms, 3))),
                "virial": numpy_to_torch(np.array([[2.0] * 9])),
                "atom_ener": numpy_to_torch(
                    np.ones((nframes, natoms)) * (1.0 + energy_diff) / natoms
                ),
                "find_energy": 1.0,
                "find_force": 0.0,  # Disable force to focus on energy/virial
                "find_force_mag": 0.0,
                "find_virial": 1.0,
                "find_atom_ener": 0.0,
            }
            return predict, label

        # Create loss functions
        loss_intensive = EnerSpinLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive_ener_virial=True,
        )
        loss_legacy = EnerSpinLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive_ener_virial=False,
        )

        # Compute losses for small system
        predict_small, label_small = create_data_with_fixed_residual(
            natoms_small, n_magnetic, fixed_energy_diff
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

        # Compute losses for large system (proportionally scale magnetic atoms)
        predict_large, label_large = create_data_with_fixed_residual(
            natoms_large, n_magnetic * 2, fixed_energy_diff
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
        """Test that intensive_ener_virial=True produces different loss than intensive_ener_virial=False."""
        if not INSTALLED_PT:
            self.skipTest("PyTorch not installed")

        rng = np.random.default_rng(20250419)
        nframes = 1
        natoms = 8
        n_magnetic = 4

        mask_mag = np.zeros((nframes, natoms, 1), dtype=bool)
        mask_mag[:, :n_magnetic, :] = True

        predict = {
            "energy": numpy_to_torch(rng.random((nframes,))),
            "force": numpy_to_torch(rng.random((nframes, natoms, 3))),
            "force_mag": numpy_to_torch(rng.random((nframes, natoms, 3))),
            "mask_mag": mask_mag,
            "virial": numpy_to_torch(rng.random((nframes, 9))),
            "atom_energy": numpy_to_torch(rng.random((nframes, natoms))),
        }
        label = {
            "energy": numpy_to_torch(rng.random((nframes,))),
            "force": numpy_to_torch(rng.random((nframes, natoms, 3))),
            "force_mag": numpy_to_torch(rng.random((nframes, natoms, 3))),
            "virial": numpy_to_torch(rng.random((nframes, 9))),
            "atom_ener": numpy_to_torch(rng.random((nframes, natoms))),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_force_mag": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 0.0,
        }

        # Create loss functions with intensive_ener_virial=True and intensive_ener_virial=False
        loss_intensive = EnerSpinLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive_ener_virial=True,
        )
        loss_legacy = EnerSpinLossPT(
            starter_learning_rate=1e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_fr=0.0,
            limit_pref_fr=0.0,
            start_pref_fm=0.0,
            limit_pref_fm=0.0,
            start_pref_v=1.0,
            limit_pref_v=1.0,
            intensive_ener_virial=False,
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
        # The intensive version should have an extra 1/N factor
        expected_ratio = 1.0 / natoms
        actual_ratio = loss_intensive_val / loss_legacy_val

        self.assertAlmostEqual(
            actual_ratio,
            expected_ratio,
            places=5,
            msg=f"Expected intensive/legacy ratio ~{expected_ratio:.6f}, got {actual_ratio:.6f}",
        )
