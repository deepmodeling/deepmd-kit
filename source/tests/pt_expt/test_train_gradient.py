# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt_expt.train.gradient import (
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
        # still clipped via the scaled reduction rather than aborting.
        p = torch.nn.Parameter(torch.zeros(128, dtype=torch.float32, device="cpu"))
        p.grad = torch.full((128,), 1.0e19, dtype=torch.float32, device="cpu")

        total_norm = clip_grad_norm_([p], max_norm=2.0)  # stable=True (default)
        clipped_norm = torch.linalg.vector_norm(p.grad.double())

        self.assertEqual(total_norm.dtype, torch.float64)
        self.assertTrue(torch.isfinite(total_norm))
        self.assertAlmostEqual(clipped_norm.item(), 2.0, places=5)

    def test_sharded_path_clips_finite_grads(self) -> None:
        # The non-stable branch (sharded DTensor grads under FSDP2) still clips
        # ordinary replicated gradients correctly.
        p = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        p.grad = torch.tensor([4.0], device="cpu")

        norm = clip_grad_norm_([p], max_norm=1.0, stable=False)

        self.assertTrue(torch.isfinite(norm))
        self.assertAlmostEqual(p.grad.item(), 1.0, places=5)

    def test_nonfinite_grad_is_deferred_to_guard(self) -> None:
        for stable in (True, False):
            for grad_value in (float("nan"), float("inf")):
                with self.subTest(stable=stable, grad_value=grad_value):
                    p = torch.nn.Parameter(torch.zeros(4, device="cpu"))
                    p.grad = torch.full((4,), grad_value, device="cpu")

                    total_norm = clip_grad_norm_([p], max_norm=1.0, stable=stable)

                    self.assertFalse(torch.isfinite(total_norm))
                    guard = NonFiniteGradGuard()
                    guard.update(total_norm)
                    with self.assertRaisesRegex(RuntimeError, "p"):
                        guard.raise_if_nonfinite(lambda: [("p", p)])


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
        # currently finite, so the message reports the deferred diagnostic state.
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(float("inf"), device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "checkpoint interval"):
            guard.raise_if_nonfinite(self._named(1.0))

    def test_resets_after_check(self) -> None:
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(float("inf"), device="cpu"))
        with self.assertRaises(RuntimeError):
            guard.raise_if_nonfinite(self._named(1.0))
        # The flag is cleared on inspection, so a later finite interval is clean.
        guard.update(torch.tensor(1.0, device="cpu"))
        guard.raise_if_nonfinite(self._named(1.0))


class TestCheckpointPublicationGuard(unittest.TestCase):
    """Publication-boundary contract for NonFiniteGradGuard (#5816)."""

    @staticmethod
    def _named(grad_value: float = 1.0):
        p = torch.nn.Parameter(torch.zeros(2, device="cpu"))
        p.grad = torch.full((2,), grad_value, device="cpu")
        return lambda: [("layer.weight", p)]

    def test_validation_best_raises_before_serialize_and_keeps_metadata(self) -> None:
        # A non-finite norm remains sticky after a later finite update, so a
        # validation-best publish before the regular interval must still abort.
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(float("nan"), device="cpu"))
        guard.update(torch.tensor(1.0, device="cpu"))

        writes: list[str] = []
        topk_committed = {"value": False}

        def ensure() -> None:
            guard.raise_if_nonfinite(self._named(float("nan")))

        def save_validation_best() -> None:
            ensure()
            writes.append("validation-best")
            topk_committed["value"] = True

        with self.assertRaises(RuntimeError):
            save_validation_best()
        self.assertEqual(writes, [])
        self.assertFalse(topk_committed["value"])

    def test_ema_validation_best_raises_before_serialize(self) -> None:
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(float("inf"), device="cpu"))
        writes: list[str] = []

        def save_ema_validation_best() -> None:
            guard.raise_if_nonfinite(self._named(1.0))
            writes.append("ema-validation-best")

        with self.assertRaises(RuntimeError):
            save_ema_validation_best()
        self.assertEqual(writes, [])

    def test_live_and_ema_regular_boundary_validates_once(self) -> None:
        guard = NonFiniteGradGuard()
        guard.update(torch.tensor(1.0, device="cpu"))
        checks = {"count": 0}
        writes: list[str] = []

        real_raise = guard.raise_if_nonfinite

        def counting_raise(named_parameters) -> None:
            checks["count"] += 1
            real_raise(named_parameters)

        guard.raise_if_nonfinite = counting_raise  # type: ignore[method-assign]

        # One logical boundary: gate once, then write live and EMA.
        guard.raise_if_nonfinite(self._named(1.0))
        writes.append("live")
        writes.append("ema")

        self.assertEqual(checks["count"], 1)
        self.assertEqual(writes, ["live", "ema"])
        # The successful boundary cleared the flag; a second gate is a no-op.
        guard.raise_if_nonfinite(self._named(1.0))
        self.assertEqual(checks["count"], 2)


if __name__ == "__main__":
    unittest.main()
