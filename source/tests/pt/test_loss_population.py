# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for PopulationLoss and PopulationModel.

Covers:
- Correct per-frame total computation (no cross-frame cancellation for
  batch_size > 1).
- Consistent prefactor scaling across loss_func choices.
- Zero loss when predictions equal labels.
- Serialize/deserialize round-trip for PopulationFittingNet.
- End-to-end: JIT save → DeepPopulation.eval() smoke test.
"""

import os
import unittest

import numpy as np
import torch

from deepmd.infer.deep_population import (
    DeepPopulation,
)
from deepmd.pt.loss.population import (
    PopulationLoss,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.population_model import (
    PopulationModel,
)
from deepmd.pt.model.task.population import (
    PopulationFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


def _make_label_and_pred(nframes: int, natoms: int, task_dim: int = 2):
    """Return (model_pred, label) dicts with random population tensors."""
    rng = torch.Generator()
    rng.manual_seed(42)
    pop_pred = torch.rand(nframes, natoms, task_dim, generator=rng)
    pop_label = torch.rand(nframes, natoms, task_dim, generator=rng)
    model_pred = {"population": pop_pred}
    label = {"atom_population": pop_label}
    return model_pred, label, pop_pred, pop_label


class FakeModel(torch.nn.Module):
    """Thin wrapper so PopulationLoss can call model(**input_dict)."""

    def __init__(self, model_pred: dict):
        super().__init__()
        self._pred = model_pred

    def forward(self, **_kwargs):
        return self._pred


class TestPopulationLossZero(unittest.TestCase):
    """Loss should be zero when predictions exactly equal labels."""

    def test_zero_loss(self):
        natoms = 4
        nframes = 2
        task_dim = 2
        pop = torch.ones(nframes, natoms, task_dim)
        model_pred = {"population": pop}
        label = {"atom_population": pop.clone()}
        loss_fn = PopulationLoss(
            loss_func="mae",
            starter_learning_rate=1.0,
        )
        model = FakeModel(model_pred)
        _, loss, _ = loss_fn({}, model, label, natoms=natoms, learning_rate=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)


class TestPopulationLossNoFrameCancellation(unittest.TestCase):
    """Cross-frame sign cancellation must NOT reduce the total-term losses.

    Frame 0 has spin_total_pred > spin_total_label by delta.
    Frame 1 has spin_total_pred < spin_total_label by delta.
    A naive batch-wide sum would give zero; per-frame summation gives 2*delta.
    """

    def _run(self, loss_func: str):
        natoms = 3
        delta = 2.0

        # Frame 0: pred alpha-beta totals = delta above label
        # Frame 1: pred alpha-beta totals = delta below label
        # Construct so spin_total error = +delta for frame 0, -delta for frame 1.
        # Use alpha=pred, beta=0 for simplicity.
        pop_pred = torch.zeros(2, natoms, 2)
        pop_label = torch.zeros(2, natoms, 2)
        # Frame 0: each atom contributes delta/natoms extra alpha → total +delta
        pop_pred[0, :, 0] = delta / natoms
        # Frame 1: label alpha is delta/natoms above pred → total -delta
        pop_label[1, :, 0] = delta / natoms

        model_pred = {"population": pop_pred}
        label = {"atom_population": pop_label}

        loss_fn = PopulationLoss(
            loss_func=loss_func,
            starter_learning_rate=1.0,
            start_pref_spin=0.0,
            limit_pref_spin=0.0,
            start_pref_spin_total=1.0,
            limit_pref_spin_total=1.0,
            start_pref_pop=0.0,
            limit_pref_pop=0.0,
            start_pref_pop_alpha_total=0.0,
            limit_pref_pop_alpha_total=0.0,
            start_pref_pop_beta_total=0.0,
            limit_pref_pop_beta_total=0.0,
        )
        model = FakeModel(model_pred)
        _, loss, more = loss_fn({}, model, label, natoms=natoms, learning_rate=1.0)
        return loss.item(), more

    def test_no_cancellation_mae(self):
        loss, _ = self._run("mae")
        # Per-frame spin_total errors are +delta and -delta.
        # Expected spin_total_loss = |+delta| + |-delta| = 2*delta.
        self.assertAlmostEqual(loss, 4.0, places=5)

    def test_no_cancellation_smooth_mae(self):
        loss, _ = self._run("smooth_mae")
        self.assertGreater(loss, 0.0)

    def test_no_cancellation_rmse(self):
        loss, _ = self._run("rmse")
        self.assertGreater(loss, 0.0)


class TestPopulationLossPrefactorScaling(unittest.TestCase):
    """Verify that loss scales with batch size for all loss_func choices."""

    def _total_loss(self, nframes: int, loss_func: str) -> float:
        natoms = 3
        task_dim = 2
        # Constant error per atom across all frames.
        pop_pred = torch.ones(nframes, natoms, task_dim)
        pop_label = torch.zeros(nframes, natoms, task_dim)
        model_pred = {"population": pop_pred}
        label = {"atom_population": pop_label}
        loss_fn = PopulationLoss(
            loss_func=loss_func,
            starter_learning_rate=1.0,
        )
        model = FakeModel(model_pred)
        _, loss, _ = loss_fn({}, model, label, natoms=natoms, learning_rate=1.0)
        return loss.item()

    def test_mae_scales_with_nframes(self):
        l1 = self._total_loss(1, "mae")
        l2 = self._total_loss(2, "mae")
        self.assertAlmostEqual(l2 / l1, 2.0, places=5)

    def test_smooth_mae_scales_with_nframes(self):
        l1 = self._total_loss(1, "smooth_mae")
        l2 = self._total_loss(2, "smooth_mae")
        self.assertAlmostEqual(l2 / l1, 2.0, places=5)

    def test_rmse_scales_with_nframes(self):
        l1 = self._total_loss(1, "rmse")
        l2 = self._total_loss(2, "rmse")
        self.assertAlmostEqual(l2 / l1, 2.0, places=5)


class TestPopulationLossMAEValue(unittest.TestCase):
    """Validate the numeric mae loss against a manual calculation."""

    def test_mae_value(self):
        natoms = 2
        nframes = 2
        task_dim = 2
        # pred = 1, label = 0 everywhere → error = 1 per element
        pop_pred = torch.ones(nframes, natoms, task_dim)
        pop_label = torch.zeros(nframes, natoms, task_dim)
        model_pred = {"population": pop_pred}
        label = {"atom_population": pop_label}

        loss_fn = PopulationLoss(
            loss_func="mae",
            starter_learning_rate=1.0,
            start_pref_spin=1.0,
            limit_pref_spin=1.0,
            start_pref_spin_total=0.0,
            limit_pref_spin_total=0.0,
            start_pref_pop=0.0,
            limit_pref_pop=0.0,
            start_pref_pop_alpha_total=0.0,
            limit_pref_pop_alpha_total=0.0,
            start_pref_pop_beta_total=0.0,
            limit_pref_pop_beta_total=0.0,
        )
        model = FakeModel(model_pred)
        _, loss, _ = loss_fn({}, model, label, natoms=natoms, learning_rate=1.0)

        # spin = alpha - beta = 1 - 1 = 0 → spin_loss = 0
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_mae_spin_value(self):
        """Spin = alpha - beta; error counted correctly."""
        natoms = 2
        pop_pred = torch.tensor([[[2.0, 1.0], [2.0, 1.0]]])  # spin = 1 per atom
        pop_label = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])  # spin = 0 per atom
        model_pred = {"population": pop_pred}
        label = {"atom_population": pop_label}

        loss_fn = PopulationLoss(
            loss_func="mae",
            starter_learning_rate=1.0,
            start_pref_spin=1.0,
            limit_pref_spin=1.0,
            start_pref_spin_total=0.0,
            limit_pref_spin_total=0.0,
            start_pref_pop=0.0,
            limit_pref_pop=0.0,
            start_pref_pop_alpha_total=0.0,
            limit_pref_pop_alpha_total=0.0,
            start_pref_pop_beta_total=0.0,
            limit_pref_pop_beta_total=0.0,
        )
        model = FakeModel(model_pred)
        _, loss, _ = loss_fn({}, model, label, natoms=natoms, learning_rate=1.0)

        # spin_pred = [1, 1], spin_label = [0, 0]
        # spin_loss = sum(|1-0|, |1-0|) / natoms = 2 / 2 = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=6)


class TestPopulationFittingNetSerialize(unittest.TestCase):
    """Serialize → deserialize round-trip for PopulationFittingNet."""

    def setUp(self) -> None:
        self.rcut = 4.0
        self.rcut_smth = 0.5
        self.sel = [46, 92, 4]  # 3 entries → DescrptSeA infers ntypes=3
        self.nt = 3
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)

    def test_serialize_deserialize(self) -> None:
        """Output of deserialized fitting net must match the original."""
        ft0 = PopulationFittingNet(
            ntypes=self.nt,
            dim_descrpt=self.dd0.dim_out,
            neuron=[16, 16],
            mixed_types=self.dd0.mixed_types(),
        ).to(env.DEVICE)
        ft1 = PopulationFittingNet.deserialize(ft0.serialize())

        natoms = 5
        nframes = 2
        atype = torch.zeros(nframes, natoms, dtype=torch.int32, device=env.DEVICE)
        descriptor = torch.rand(
            nframes, natoms, self.dd0.dim_out, dtype=dtype, device=env.DEVICE
        )
        ret0 = ft0(descriptor, atype)
        ret1 = ft1(descriptor, atype)
        np.testing.assert_allclose(
            to_numpy_array(ret0["population"]),
            to_numpy_array(ret1["population"]),
        )


class TestPopulationModelInfer(unittest.TestCase):
    """JIT save → DeepPopulation.eval() smoke test (mirrors TestPropertyModel)."""

    def setUp(self) -> None:
        self.natoms = 5
        self.rcut = 4.0
        self.nt = 3  # must match len(sel)
        self.rcut_smth = 0.5
        self.sel = [46, 92, 4]  # 3 entries → DescrptSeA infers ntypes=3
        self.nf = 1
        self.coord = 2 * torch.rand([self.natoms, 3], dtype=dtype, device="cpu")
        cell = torch.rand([3, 3], dtype=dtype, device="cpu")
        self.cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        self.atype = torch.tensor([0, 0, 0, 1, 2], dtype=torch.int32, device="cpu")
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.ft0 = PopulationFittingNet(
            ntypes=self.nt,
            dim_descrpt=self.dd0.dim_out,
            neuron=[16, 16],
            mixed_types=self.dd0.mixed_types(),
        ).to(env.DEVICE)
        self.type_map = ["O", "H", "B"]
        self.model = PopulationModel(self.dd0, self.ft0, self.type_map)
        self.file_path = "test_population_model.pth"

    def test_deeppopulation_eval(self) -> None:
        """Save model as JIT, load as DeepPopulation, run eval."""
        coord = self.coord.reshape(self.nf, self.natoms, 3).numpy()
        cell = self.cell.reshape(self.nf, 9).numpy()
        atype = self.atype.numpy()

        jit_md = torch.jit.script(self.model)
        torch.jit.save(jit_md, self.file_path)

        load_md = DeepPopulation(self.file_path)
        (populations,) = load_md.eval(
            coords=coord, atom_types=atype, cells=cell, atomic=True
        )
        self.assertEqual(populations.shape, (self.nf, self.natoms, 2))

    def tearDown(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


if __name__ == "__main__":
    unittest.main()
