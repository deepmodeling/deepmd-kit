# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the aparam gradient supervision in EnergyStdLoss."""

import unittest

import numpy as np
import torch

from deepmd.pt.loss import (
    EnergyStdLoss,
)
from deepmd.pt.utils.env import (
    DEVICE,
)


class EnergyModelWithAparamGrad(torch.nn.Module):
    """A minimal differentiable model: energy = sum_i (W @ aparam_i).

    This ensures d(energy)/d(aparam) = W (per atom), giving us a known
    analytical gradient to validate against.
    """

    def __init__(self, numb_aparam: int) -> None:
        super().__init__()
        # A fixed weight matrix so the gradient is deterministic
        self.weight = torch.nn.Parameter(
            torch.ones(1, numb_aparam, dtype=torch.float64, device=DEVICE)
        )

    def forward(
        self,
        coord: torch.Tensor | None = None,
        atype: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        # aparam: [nf, nloc, numb_aparam]
        assert aparam is not None
        nf, nloc, _ = aparam.shape
        # atom_energy: [nf, nloc] = sum over aparam dim weighted by self.weight
        atom_energy = torch.sum(aparam * self.weight, dim=-1)
        # energy: [nf, 1]
        energy = atom_energy.sum(dim=-1, keepdim=True)
        # force: [nf, nloc, 3] — dummy zeros
        force = torch.zeros(nf, nloc, 3, dtype=aparam.dtype, device=DEVICE)
        return {
            "energy": energy,
            "atom_energy": atom_energy,
            "force": force,
        }


class TestGradAparamLoss(unittest.TestCase):
    """Test aparam gradient loss computation."""

    def setUp(self) -> None:
        self.nf = 2  # number of frames
        self.nloc = 4  # number of atoms
        self.numb_aparam = 3
        self.start_lr = 1.0
        self.cur_lr = 1.0
        self.rng = np.random.default_rng(42)

    def _make_loss(self, **kwargs) -> EnergyStdLoss:
        """Create an EnergyStdLoss with aparam gradient enabled."""
        defaults = {
            "starter_learning_rate": self.start_lr,
            "start_pref_e": 0.0,
            "limit_pref_e": 0.0,
            "start_pref_f": 0.0,
            "limit_pref_f": 0.0,
            "start_pref_ap": 1.0,
            "limit_pref_ap": 1.0,
            "numb_aparam": self.numb_aparam,
        }
        defaults.update(kwargs)
        return EnergyStdLoss(**defaults)

    def _make_data(self):
        """Create random aparam input and grad_aparam label."""
        aparam = self.rng.random((self.nf, self.nloc, self.numb_aparam)).astype(
            np.float64
        )
        grad_aparam_label = self.rng.random(
            (self.nf, self.nloc, self.numb_aparam)
        ).astype(np.float64)
        return aparam, grad_aparam_label

    def test_loss_nonzero_and_backward(self) -> None:
        """Loss should be > 0 and backward should flow gradients to model params."""
        loss_fn = self._make_loss()
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        aparam_np, grad_ap_label_np = self._make_data()

        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": torch.from_numpy(aparam_np),
        }
        label = {
            "grad_aparam": torch.from_numpy(grad_ap_label_np),
            "find_grad_aparam": 1.0,
        }

        model_pred, loss, more_loss = loss_fn(
            input_dict, model, label, self.nloc, self.cur_lr
        )

        # loss should be a positive scalar
        self.assertGreater(loss.item(), 0.0)

        # more_loss should contain the expected keys
        self.assertIn("rmse_grad_aparam", more_loss)
        self.assertIn("l2_grad_aparam_loss", more_loss)

        # backward should work (gradients flow to model.weight)
        loss.backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertFalse(torch.all(model.weight.grad == 0.0))

    def test_analytical_gradient(self) -> None:
        """Verify autograd computes the correct gradient analytically.

        For EnergyModelWithAparamGrad, d(energy)/d(aparam_i) = weight for every atom.
        """
        loss_fn = self._make_loss()
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        aparam_np, _ = self._make_data()

        # The analytical gradient is the model weight broadcast to all atoms
        expected_grad = model.weight.detach().numpy()  # [1, numb_aparam]
        expected_grad_full = np.broadcast_to(
            expected_grad, (self.nf, self.nloc, self.numb_aparam)
        )

        # Use expected gradient as label => loss should be 0
        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": torch.from_numpy(aparam_np),
        }
        label = {
            "grad_aparam": torch.from_numpy(expected_grad_full.copy()),
            "find_grad_aparam": 1.0,
        }

        _, loss, more_loss = loss_fn(input_dict, model, label, self.nloc, self.cur_lr)

        # loss should be essentially zero since label matches analytical grad
        self.assertAlmostEqual(loss.item(), 0.0, places=10)
        self.assertAlmostEqual(more_loss["rmse_grad_aparam"].item(), 0.0, places=10)

    def test_absent_label_no_crash(self) -> None:
        """When grad_aparam label is missing, loss should still work (skip aparam term).

        aparam_grad is still computed and placed in model_pred even without labels.
        """
        loss_fn = self._make_loss()
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        aparam_np, _ = self._make_data()

        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": torch.from_numpy(aparam_np),
        }
        # No grad_aparam in label
        label = {}

        model_pred, loss, more_loss = loss_fn(
            input_dict, model, label, self.nloc, self.cur_lr
        )

        # All prefactors for e/f/v are 0, and no grad_aparam label => loss = 0
        self.assertEqual(loss.item(), 0.0)
        self.assertNotIn("rmse_grad_aparam", more_loss)
        # aparam_grad should still be computed and available in model_pred
        self.assertIn("aparam_grad", model_pred)
        self.assertEqual(
            model_pred["aparam_grad"].shape,
            (self.nf, self.nloc, self.numb_aparam),
        )

    def test_disabled_by_default(self) -> None:
        """When start_pref_ap=0 and limit_pref_ap=0 (default), no aparam grad computed."""
        loss_fn = EnergyStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=0.02,
            limit_pref_e=1.0,
            start_pref_f=1000.0,
            limit_pref_f=1.0,
        )
        # has_ap should be False
        self.assertFalse(loss_fn.has_ap)

    def test_error_on_zero_numb_aparam(self) -> None:
        """Enabling aparam loss with numb_aparam=0 should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            EnergyStdLoss(
                starter_learning_rate=self.start_lr,
                start_pref_ap=1.0,
                limit_pref_ap=1.0,
                numb_aparam=0,
            )

    def test_label_requirement_includes_grad_aparam(self) -> None:
        """label_requirement should include grad_aparam when aparam loss is enabled."""
        loss_fn = self._make_loss()
        names = [req.key for req in loss_fn.label_requirement]
        self.assertIn("grad_aparam", names)

    def test_label_requirement_excludes_grad_aparam_when_disabled(self) -> None:
        """label_requirement should NOT include grad_aparam when disabled."""
        loss_fn = EnergyStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=0.02,
            limit_pref_e=1.0,
        )
        names = [req.key for req in loss_fn.label_requirement]
        self.assertNotIn("grad_aparam", names)

    def test_serialize_roundtrip(self) -> None:
        """Serialization should preserve start_pref_ap and limit_pref_ap."""
        loss_fn = self._make_loss(start_pref_ap=100.0, limit_pref_ap=1.0)
        data = loss_fn.serialize()
        self.assertEqual(data["start_pref_ap"], 100.0)
        self.assertEqual(data["limit_pref_ap"], 1.0)

    def test_pref_schedule(self) -> None:
        """The aparam loss prefactor should interpolate between start and limit."""
        loss_fn = self._make_loss(
            starter_learning_rate=1.0,
            start_pref_ap=100.0,
            limit_pref_ap=1.0,
        )
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        aparam_np, grad_ap_label_np = self._make_data()
        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": torch.from_numpy(aparam_np),
        }
        label = {
            "grad_aparam": torch.from_numpy(grad_ap_label_np),
            "find_grad_aparam": 1.0,
        }

        # At cur_lr == start_lr, coef=1.0, pref_ap = start_pref_ap
        _, loss_start, _ = loss_fn(input_dict, model, label, self.nloc, 1.0)

        # At cur_lr = 0.0, coef=0.0, pref_ap = limit_pref_ap
        _, loss_limit, _ = loss_fn(input_dict, model, label, self.nloc, 0.0)

        # start_pref_ap (100) >> limit_pref_ap (1), so loss at start should be larger
        self.assertGreater(loss_start.item(), loss_limit.item())

    def test_combined_energy_and_aparam_loss(self) -> None:
        """Energy and aparam gradient losses should combine correctly."""
        loss_fn = self._make_loss(
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_ap=1.0,
            limit_pref_ap=1.0,
        )
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        aparam_np, grad_ap_label_np = self._make_data()
        energy_label = self.rng.random((self.nf, 1)).astype(np.float64)

        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": torch.from_numpy(aparam_np),
        }
        label = {
            "energy": torch.from_numpy(energy_label),
            "find_energy": 1.0,
            "grad_aparam": torch.from_numpy(grad_ap_label_np),
            "find_grad_aparam": 1.0,
        }

        model_pred, loss, more_loss = loss_fn(
            input_dict, model, label, self.nloc, self.cur_lr
        )

        self.assertGreater(loss.item(), 0.0)
        # Both energy and aparam gradient losses should be present
        self.assertIn("rmse_e", more_loss)
        self.assertIn("rmse_grad_aparam", more_loss)

        # backward should work
        loss.backward()
        self.assertIsNotNone(model.weight.grad)

    def test_no_aparam_in_input(self) -> None:
        """When aparam is None in input_dict, aparam grad is skipped gracefully."""
        loss_fn = self._make_loss()
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        # Override model forward to work without aparam
        original_forward = model.forward

        def forward_no_aparam(**kwargs):
            kwargs["aparam"] = torch.zeros(
                self.nf, self.nloc, self.numb_aparam, dtype=torch.float64, device=DEVICE
            )
            return original_forward(**kwargs)

        model.forward = forward_no_aparam

        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": None,  # No aparam
        }
        label = {
            "grad_aparam": torch.randn(
                self.nf, self.nloc, self.numb_aparam, dtype=torch.float64, device=DEVICE
            ),
            "find_grad_aparam": 1.0,
        }

        # Should not crash — ap_for_grad is None when aparam is None
        model_pred, loss, more_loss = loss_fn(
            input_dict, model, label, self.nloc, self.cur_lr
        )
        self.assertNotIn("rmse_grad_aparam", more_loss)

    def test_inference_no_grad_context(self) -> None:
        """aparam_grad should be computed inside torch.no_grad() (inference mode).

        model_pred must contain 'aparam_grad', and rmse metric should be present
        when labels are provided.
        """
        loss_fn = self._make_loss()
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        aparam_np, grad_ap_label_np = self._make_data()

        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": torch.from_numpy(aparam_np),
        }
        label = {
            "grad_aparam": torch.from_numpy(grad_ap_label_np),
            "find_grad_aparam": 1.0,
        }

        with torch.no_grad():
            model_pred, loss, more_loss = loss_fn(
                input_dict, model, label, self.nloc, self.cur_lr
            )

        # aparam_grad should be in model_pred even in no_grad context
        self.assertIn("aparam_grad", model_pred)
        self.assertEqual(
            model_pred["aparam_grad"].shape,
            (self.nf, self.nloc, self.numb_aparam),
        )
        # The gradient should be detached (no computation graph) in inference
        self.assertFalse(model_pred["aparam_grad"].requires_grad)

        # RMSE metric should still be present
        self.assertIn("rmse_grad_aparam", more_loss)

    def test_inference_analytical_gradient_correctness(self) -> None:
        """Verify gradient correctness is preserved during torch.no_grad() inference."""
        loss_fn = self._make_loss()
        model = EnergyModelWithAparamGrad(self.numb_aparam).double()

        aparam_np, _ = self._make_data()
        # Analytical gradient = weight broadcast to all atoms
        expected_grad = model.weight.detach().numpy()
        expected_grad_full = np.broadcast_to(
            expected_grad, (self.nf, self.nloc, self.numb_aparam)
        ).copy()

        input_dict = {
            "coord": None,
            "atype": None,
            "box": None,
            "fparam": None,
            "aparam": torch.from_numpy(aparam_np),
        }
        label = {
            "grad_aparam": torch.from_numpy(expected_grad_full),
            "find_grad_aparam": 1.0,
        }

        with torch.no_grad():
            model_pred, loss, more_loss = loss_fn(
                input_dict, model, label, self.nloc, self.cur_lr
            )

        # aparam_grad in model_pred should match analytical gradient
        np.testing.assert_allclose(
            model_pred["aparam_grad"].numpy(),
            expected_grad_full,
            atol=1e-10,
        )
        # Loss should be zero when label matches analytical gradient
        self.assertAlmostEqual(loss.item(), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
