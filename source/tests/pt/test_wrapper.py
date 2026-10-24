# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for PyTorch model wrapper behavior."""

from __future__ import (
    annotations,
)

import unittest

import torch

from deepmd.pt.train.wrapper import (
    ModelWrapper,
)


class _LinearToyModel(torch.nn.Module):
    def __init__(self, *, fail_forward: bool = False) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 1, bias=False, device="cpu")
        self.scale = torch.nn.Parameter(torch.ones((), device="cpu"))
        self.fail_forward = fail_forward
        self.last_requires_grad: tuple[bool, ...] | None = None
        self.last_charge_spin: torch.Tensor | None = None

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del atype, box, do_atomic_virial, fparam, aparam
        self.last_charge_spin = charge_spin
        self.last_requires_grad = tuple(
            param.requires_grad for param in self.parameters()
        )
        if self.fail_forward:
            raise RuntimeError("intentional toy failure")
        coord_req = coord.clone().requires_grad_(True)
        atom_energy = self.scale * self.linear(coord_req).sum(dim=1, keepdim=True)
        energy = atom_energy.sum(dim=1)
        force = -torch.autograd.grad(energy.sum(), coord_req, create_graph=True)[0]
        return {
            "atom_energy": atom_energy,
            "energy": energy,
            "force": force,
        }


class _LegacyToyModel(torch.nn.Module):
    """Model with the scripted signature used before charge_spin was added."""

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a simple energy without accepting a charge_spin keyword."""
        return {"energy": coord.sum(dim=-1).sum(dim=-1, keepdim=True)}


class _EnergyLoss(torch.nn.Module):
    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: torch.Tensor | None,
        natoms: int,
        learning_rate: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
        del label, natoms, learning_rate
        pred = model(**input_dict)
        loss = pred["energy"].sum()
        return pred, loss, {}


class TestModelWrapper(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20240611)
        self.coord = torch.randn(2, 5, 3, device="cpu")
        self.atype = torch.zeros(2, 5, dtype=torch.long, device="cpu")

    def test_legacy_scripted_signature_without_charge_spin(self) -> None:
        """Omitting charge_spin must work without a neighbor-list dependency."""
        for mode in ("inference", "skip_loss", "training"):
            with self.subTest(mode=mode):
                model = torch.jit.script(_LegacyToyModel())
                loss = None if mode == "inference" else _EnergyLoss()
                wrapper = ModelWrapper(model, loss)
                expected = model(self.coord, self.atype)
                actual, _, _ = wrapper(
                    self.coord, self.atype, skip_loss=mode == "skip_loss"
                )
                torch.testing.assert_close(actual["energy"], expected["energy"])

    def test_supplied_charge_spin_is_forwarded_unchanged(self) -> None:
        """Forward the supplied tensor through inference and loss evaluation."""
        charge_spin = torch.tensor([[1.0, 0.0], [-1.0, 2.0]], device="cpu")
        for mode in ("inference", "skip_loss", "training"):
            with self.subTest(mode=mode):
                model = _LinearToyModel()
                loss = None if mode == "inference" else _EnergyLoss()
                wrapper = ModelWrapper(model, loss)
                wrapper(
                    self.coord,
                    self.atype,
                    skip_loss=mode == "skip_loss",
                    charge_spin=charge_spin,
                )
                self.assertIs(model.last_charge_spin, charge_spin)

    def test_inference_wrapper_freezes_parameters_without_changing_predictions(
        self,
    ) -> None:
        model = _LinearToyModel()
        reference_model = _LinearToyModel()
        reference_model.load_state_dict(model.state_dict())
        wrapper = ModelWrapper(model)
        reference_wrapper = ModelWrapper(reference_model, _EnergyLoss())

        ref, _, _ = reference_wrapper(self.coord, self.atype, skip_loss=True)
        out, _, _ = wrapper(self.coord, self.atype)

        self.assertEqual(model.last_requires_grad, (False, False))
        self.assertEqual(reference_model.last_requires_grad, (True, True))
        self.assertTrue(all(param.requires_grad for param in wrapper.parameters()))
        for key in ("atom_energy", "energy", "force"):
            torch.testing.assert_close(out[key], ref[key])

    def test_inference_wrapper_restores_mixed_parameter_flags(self) -> None:
        model = _LinearToyModel()
        model.linear.weight.requires_grad_(False)
        wrapper = ModelWrapper(model)

        wrapper(self.coord, self.atype)

        self.assertEqual(model.last_requires_grad, (False, False))
        self.assertFalse(model.linear.weight.requires_grad)
        self.assertTrue(model.scale.requires_grad)

    def test_inference_wrapper_restores_parameters_after_exception(self) -> None:
        model = _LinearToyModel(fail_forward=True)
        wrapper = ModelWrapper(model)

        with self.assertRaisesRegex(RuntimeError, "intentional toy failure"):
            wrapper(self.coord, self.atype)

        self.assertEqual(model.last_requires_grad, (False, False))
        self.assertTrue(all(param.requires_grad for param in wrapper.parameters()))

    def test_multitask_inference_wrapper_freezes_selected_head(self) -> None:
        model_a = _LinearToyModel()
        model_b = _LinearToyModel()
        wrapper = ModelWrapper({"a": model_a, "b": model_b})

        wrapper(self.coord, self.atype, task_key="b")

        self.assertIsNone(model_a.last_requires_grad)
        self.assertEqual(model_b.last_requires_grad, (False, False))
        self.assertTrue(all(param.requires_grad for param in wrapper.parameters()))

    def test_skip_loss_keeps_training_gradients(self) -> None:
        model = _LinearToyModel()
        wrapper = ModelWrapper(model, _EnergyLoss())

        pred, _, _ = wrapper(self.coord, self.atype, skip_loss=True)
        pred["energy"].sum().backward()

        self.assertEqual(model.last_requires_grad, (True, True))
        self.assertIsNotNone(model.linear.weight.grad)
        self.assertIsNotNone(model.scale.grad)


if __name__ == "__main__":
    unittest.main()
