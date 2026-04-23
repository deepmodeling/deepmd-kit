# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the use_default_pf feature in EnergyStdLoss (PT-only, no TF dependency)."""

import unittest
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.pt.loss import (
    EnergyStdLoss,
)
from deepmd.pt.utils import (
    dp_random,
    env,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from ..seed import (
    GLOBAL_SEED,
)

energy_data_requirement = [
    DataRequirementItem(
        "energy",
        ndof=1,
        atomic=False,
        must=False,
        high_prec=True,
    ),
    DataRequirementItem(
        "force",
        ndof=3,
        atomic=True,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "virial",
        ndof=9,
        atomic=False,
        must=False,
        high_prec=False,
    ),
]


def get_single_batch(dataset, index=None):
    if index is None:
        index = dp_random.choice(np.arange(len(dataset)))
    np_batch = dataset[index]
    pt_batch = {}
    for key in ["coord", "box", "force", "energy", "virial", "atype", "natoms"]:
        if key in np_batch.keys():
            np_batch[key] = np.expand_dims(np_batch[key], axis=0)
            pt_batch[key] = torch.as_tensor(np_batch[key], device=env.DEVICE)
            if key in ["coord", "force"]:
                np_batch[key] = np_batch[key].reshape(1, -1)
    return np_batch, pt_batch


def get_batch(system, type_map, data_requirement):
    dataset = DeepmdDataSetForLoader(system, type_map)
    dataset.add_data_requirement(data_requirement)
    np_batch, pt_batch = get_single_batch(dataset)
    return np_batch, pt_batch


class TestEnerStdLossDefaultPf(unittest.TestCase):
    """Test use_default_pf feature in EnergyStdLoss."""

    def setUp(self) -> None:
        self.start_lr = 1.1
        self.cur_lr = 1.2
        self.start_pref_e = 0.02
        self.limit_pref_e = 1.0
        self.start_pref_f = 0.0
        self.limit_pref_f = 0.0
        self.start_pref_v = 0.0
        self.limit_pref_v = 0.0
        self.start_pref_pf = 1.0
        self.limit_pref_pf = 1.0

        self.system = str(Path(__file__).parent / "water/data/data_0")
        self.type_map = ["H", "O"]

        np_batch, pt_batch = get_batch(
            self.system, self.type_map, energy_data_requirement
        )
        natoms = np_batch["natoms"]
        self.nloc = int(natoms[0][0])
        rng = np.random.default_rng(GLOBAL_SEED)

        l_energy, l_force, l_virial = (
            np_batch["energy"],
            np_batch["force"],
            np_batch["virial"],
        )
        p_energy, p_force, p_virial = (
            np.ones_like(l_energy),
            np.ones_like(l_force),
            np.ones_like(l_virial),
        )
        nloc = self.nloc
        batch_size = pt_batch["coord"].shape[0]
        p_atom_energy = rng.random(size=[batch_size, nloc])
        atom_pref = np.ones([batch_size, nloc * 3])

        self.model_pred = {
            "energy": torch.from_numpy(p_energy),
            "force": torch.from_numpy(p_force),
            "virial": torch.from_numpy(p_virial),
            "atom_energy": torch.from_numpy(p_atom_energy),
        }
        # label WITH find_atom_pref (simulates data with atom_pref.npy)
        self.label_with_pref = {
            "energy": torch.from_numpy(l_energy),
            "find_energy": 1.0,
            "force": torch.from_numpy(l_force),
            "find_force": 1.0,
            "virial": torch.from_numpy(l_virial),
            "find_virial": 0.0,
            "atom_pref": torch.from_numpy(atom_pref),
            "find_atom_pref": 1.0,
        }
        # label WITHOUT find_atom_pref (simulates data without atom_pref.npy)
        self.label_without_pref = {
            "energy": torch.from_numpy(l_energy),
            "find_energy": 1.0,
            "force": torch.from_numpy(l_force),
            "find_force": 1.0,
            "virial": torch.from_numpy(l_virial),
            "find_virial": 0.0,
            "atom_pref": torch.from_numpy(atom_pref),
            "find_atom_pref": 0.0,
        }
        self.natoms = pt_batch["natoms"]

    def test_default_pf_enabled(self) -> None:
        """With use_default_pf=True, pf loss should be computed even without find_atom_pref."""
        loss_fn = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            start_pref_pf=self.start_pref_pf,
            limit_pref_pf=self.limit_pref_pf,
            use_default_pf=True,
        )

        def fake_model():
            return self.model_pred

        # With find_atom_pref=0.0 but use_default_pf=True, pf loss should still be computed
        _, pt_loss, pt_more_loss = loss_fn(
            {},
            fake_model,
            self.label_without_pref,
            self.nloc,
            self.cur_lr,
        )
        pt_loss_val = pt_loss.detach().cpu().numpy()
        # loss should be non-zero because pf loss is activated via use_default_pf
        self.assertTrue(pt_loss_val != 0.0)
        self.assertIn("rmse_pf", pt_more_loss)
        # The pref_force_loss should be a valid number (not NaN)
        self.assertFalse(np.isnan(pt_more_loss["l2_pref_force_loss"]))

    def test_default_pf_disabled(self) -> None:
        """With use_default_pf=False (default), pf loss should NOT be computed without find_atom_pref."""
        loss_fn = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            start_pref_pf=self.start_pref_pf,
            limit_pref_pf=self.limit_pref_pf,
            use_default_pf=False,
        )

        def fake_model():
            return self.model_pred

        # With find_atom_pref=0.0 and use_default_pf=False, pf loss contribution is zero
        _, pt_loss_without, pt_more_loss_without = loss_fn(
            {},
            fake_model,
            self.label_without_pref,
            self.nloc,
            self.cur_lr,
        )
        # With find_atom_pref=1.0, pf loss should be computed
        _, pt_loss_with, pt_more_loss_with = loss_fn(
            {},
            fake_model,
            self.label_with_pref,
            self.nloc,
            self.cur_lr,
        )
        # without find_atom_pref, the pf part contributes nothing
        self.assertTrue(np.isnan(pt_more_loss_without["l2_pref_force_loss"]))
        # with find_atom_pref, pf loss should be computed
        self.assertFalse(np.isnan(pt_more_loss_with["l2_pref_force_loss"]))

    def test_default_pf_consistency(self) -> None:
        """With use_default_pf=True and atom_pref=1.0, results should match explicit find_atom_pref=1.0."""
        loss_fn_default = EnergyStdLoss(
            self.start_lr,
            self.start_pref_e,
            self.limit_pref_e,
            self.start_pref_f,
            self.limit_pref_f,
            self.start_pref_v,
            self.limit_pref_v,
            start_pref_pf=self.start_pref_pf,
            limit_pref_pf=self.limit_pref_pf,
            use_default_pf=True,
        )

        def fake_model():
            return self.model_pred

        # use_default_pf=True with find_atom_pref=0.0
        _, pt_loss_default, _ = loss_fn_default(
            {},
            fake_model,
            self.label_without_pref,
            self.nloc,
            self.cur_lr,
        )
        # use_default_pf=True with find_atom_pref=1.0 (should also give same result)
        _, pt_loss_explicit, _ = loss_fn_default(
            {},
            fake_model,
            self.label_with_pref,
            self.nloc,
            self.cur_lr,
        )
        # Both should be the same since use_default_pf overrides find_atom_pref
        self.assertTrue(
            np.allclose(
                pt_loss_default.detach().cpu().numpy(),
                pt_loss_explicit.detach().cpu().numpy(),
            )
        )

    def test_label_requirement_force_included(self) -> None:
        """When has_pf=True but has_f=False, force should still be in label_requirement."""
        loss_fn = EnergyStdLoss(
            self.start_lr,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_pf=self.start_pref_pf,
            limit_pref_pf=self.limit_pref_pf,
            use_default_pf=True,
        )
        label_req = loss_fn.label_requirement
        keys = [r.key for r in label_req]
        self.assertIn("force", keys)
        self.assertIn("atom_pref", keys)

    def test_label_requirement_atom_pref_default(self) -> None:
        """atom_pref DataRequirementItem should have default=1.0."""
        loss_fn = EnergyStdLoss(
            self.start_lr,
            start_pref_pf=self.start_pref_pf,
            limit_pref_pf=self.limit_pref_pf,
            use_default_pf=True,
        )
        label_req = loss_fn.label_requirement
        atom_pref_req = next(r for r in label_req if r.key == "atom_pref")
        self.assertEqual(atom_pref_req.default, 1.0)

    def test_serialize_deserialize(self) -> None:
        """Serialization round-trip should preserve use_default_pf."""
        loss_fn = EnergyStdLoss(
            self.start_lr,
            start_pref_pf=self.start_pref_pf,
            limit_pref_pf=self.limit_pref_pf,
            use_default_pf=True,
        )
        data = loss_fn.serialize()
        self.assertTrue(data["use_default_pf"])
        self.assertEqual(data["@version"], 3)

        loss_fn2 = EnergyStdLoss.deserialize(data)
        self.assertTrue(loss_fn2.use_default_pf)


if __name__ == "__main__":
    unittest.main()
