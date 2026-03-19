# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestEnergyLossBase(unittest.TestCase):
    """Base class providing common setup for dpmodel EnergyLoss tests."""

    def _make_data(self, natoms=5, nframes=2, numb_generalized_coord=0):
        """Generate fake model predictions and labels."""
        rng = np.random.default_rng(GLOBAL_SEED)
        model_dict = {
            "energy": rng.random((nframes, 1)),
            "force": rng.random((nframes, natoms, 3)),
            "virial": rng.random((nframes, 9)),
            "atom_energy": rng.random((nframes, natoms, 1)),
        }
        label_dict = {
            "energy": rng.random((nframes, 1)),
            "force": rng.random((nframes, natoms, 3)),
            "virial": rng.random((nframes, 9)),
            "atom_ener": rng.random((nframes, natoms, 1)),
            "atom_pref": rng.random((nframes, natoms * 3)),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 1.0,
            "find_atom_pref": 1.0,
        }
        if numb_generalized_coord > 0:
            label_dict["drdq"] = rng.random(
                (nframes, natoms * 3 * numb_generalized_coord)
            )
            label_dict["find_drdq"] = 1.0
        if hasattr(self, "enable_atom_ener_coeff") and self.enable_atom_ener_coeff:
            label_dict["atom_ener_coeff"] = rng.random((nframes, natoms, 1))
        return model_dict, label_dict, natoms


class TestEnergyLossBasic(TestEnergyLossBase):
    """Test basic energy loss (e, f, v, ae)."""

    def test_forward(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            start_pref_ae=1.0,
            limit_pref_ae=0.5,
        )
        model_dict, label_dict, natoms = self._make_data()
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)
        self.assertIn("rmse_e", more_loss)
        self.assertIn("rmse_f", more_loss)
        self.assertIn("rmse_v", more_loss)
        self.assertIn("rmse_ae", more_loss)


class TestEnergyLossAecoeff(TestEnergyLossBase):
    """Test energy loss with atom_ener_coeff."""

    enable_atom_ener_coeff = True

    def test_forward(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            enable_atom_ener_coeff=True,
        )
        model_dict, label_dict, natoms = self._make_data()
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)


class TestEnergyLossGeneralizedForce(TestEnergyLossBase):
    """Test energy loss with generalized force (numb_generalized_coord > 0).

    This exercises the code path with natoms used as int scalar
    (not array), which previously had a natoms[0] bug.
    """

    def test_forward(self) -> None:
        numb_generalized_coord = 2
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            start_pref_ae=1.0,
            limit_pref_ae=0.5,
            start_pref_pf=1.0,
            limit_pref_pf=0.5,
            start_pref_gf=1.0,
            limit_pref_gf=0.5,
            numb_generalized_coord=numb_generalized_coord,
        )
        model_dict, label_dict, natoms = self._make_data(
            numb_generalized_coord=numb_generalized_coord,
        )
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)
        self.assertIn("rmse_gf", more_loss)
        self.assertIn("rmse_pf", more_loss)


class TestEnergyLossHuber(TestEnergyLossBase):
    """Test energy loss with Huber loss."""

    def test_forward(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            use_huber=True,
            huber_delta=0.01,
        )
        model_dict, label_dict, natoms = self._make_data()
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)


class TestEnergyLossSerialize(TestEnergyLossBase):
    """Test serialize/deserialize round-trip."""

    def test_serialize_deserialize(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            start_pref_gf=1.0,
            limit_pref_gf=0.5,
            numb_generalized_coord=2,
        )
        data = loss_fn.serialize()
        loss_fn2 = EnergyLoss.deserialize(data)
        model_dict, label_dict, natoms = self._make_data(numb_generalized_coord=2)
        loss1, more1 = loss_fn.call(1.0, natoms, model_dict, label_dict)
        loss2, more2 = loss_fn2.call(1.0, natoms, model_dict, label_dict)
        np.testing.assert_allclose(loss1, loss2)
        for key in more1:
            np.testing.assert_allclose(more1[key], more2[key])


if __name__ == "__main__":
    unittest.main()
