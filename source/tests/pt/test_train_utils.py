# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.train.utils import (
    NonFiniteGradGuard,
    clip_grad_norm_,
)


class TestClipGradNorm(unittest.TestCase):
    def test_clips_to_max_norm(self) -> None:
        p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        p.grad = torch.tensor([2.0], device="cpu")
        norm = clip_grad_norm_([p], max_norm=1.0)

        self.assertTrue(torch.isfinite(norm))
        self.assertAlmostEqual(p.grad.item(), 1.0, places=5)

    def test_leaves_small_gradients_unchanged(self) -> None:
        p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        p.grad = torch.tensor([0.5], device="cpu")
        norm = clip_grad_norm_([p], max_norm=1.0)

        self.assertTrue(torch.isfinite(norm))
        self.assertAlmostEqual(p.grad.item(), 0.5, places=5)

    def test_empty_parameters(self) -> None:
        norm = clip_grad_norm_([], max_norm=1.0)

        self.assertEqual(norm.item(), 0.0)

    def test_stable_norm_survives_float32_overflow(self) -> None:
        # A single parameter whose gradient sum-of-squares overflows float32 is
        # still clipped via the float64 reduction rather than aborting.
        p = torch.nn.Parameter(torch.zeros(128, dtype=torch.float32, device="cpu"))
        p.grad = torch.full((128,), 1.0e19, dtype=torch.float32, device="cpu")

        total_norm = clip_grad_norm_([p], max_norm=2.0)  # stable=True (default)
        clipped_norm = torch.linalg.vector_norm(p.grad.double())

        self.assertEqual(total_norm.dtype, torch.float64)
        self.assertTrue(torch.isfinite(total_norm))
        self.assertAlmostEqual(clipped_norm.item(), 2.0, places=3)

    def test_sharded_path_clips_finite_grads(self) -> None:
        # The non-stable branch (sharded DTensor grads under FSDP2) still clips
        # ordinary replicated gradients correctly.
        p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        p.grad = torch.tensor([4.0], device="cpu")

        norm = clip_grad_norm_([p], max_norm=1.0, stable=False)

        self.assertTrue(torch.isfinite(norm))
        self.assertAlmostEqual(p.grad.item(), 1.0, places=5)


class TestNonFiniteGradGuard(unittest.TestCase):
    @staticmethod
    def _named(grad_value: float):
        p = torch.nn.Parameter(torch.zeros(2, device="cpu"))
        p.grad = torch.full((2,), grad_value, device="cpu")
        return lambda: [("layer.weight", p)]

    def test_finite_norms_do_not_raise(self) -> None:
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(1.0, device="cpu"))
        guard.update(torch.tensor(3.0, device="cpu"))
        guard.raise_if_nonfinite(self._named(1.0))

    def test_no_update_is_noop(self) -> None:
        NonFiniteGradGuard().raise_if_nonfinite(self._named(1.0))

    def test_reports_offending_parameter(self) -> None:
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(float("nan"), device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "layer.weight"):
            guard.raise_if_nonfinite(self._named(float("nan")))

    def test_reports_reduction_overflow(self) -> None:
        # The norm was flagged non-finite, yet every individual gradient is
        # finite, so the message attributes it to the reduction.
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(float("inf"), device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "norm reduction"):
            guard.raise_if_nonfinite(self._named(1.0))

    def test_resets_after_check(self) -> None:
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(float("inf"), device="cpu"))
        with self.assertRaises(RuntimeError):
            guard.raise_if_nonfinite(self._named(1.0))
        # The flag is cleared on inspection, so a later finite interval is clean.
        guard.update(torch.tensor(1.0, device="cpu"))
        guard.raise_if_nonfinite(self._named(1.0))


if __name__ == "__main__":
    unittest.main()
