# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

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
    """Publication-boundary contract via real Trainer save entry points (#5816)."""

    @staticmethod
    def _named_parameters(grad_value: float = 1.0):
        p = torch.nn.Parameter(torch.zeros(2, device="cpu"))
        p.grad = torch.full((2,), grad_value, device="cpu")
        return lambda: [("layer.weight", p)]

    def _bare_trainer(self, *, with_ema: bool = False):
        from types import SimpleNamespace

        from deepmd.pt_expt.train.training import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.nonfinite_grad_guard = NonFiniteGradGuard()
        trainer.wrapper = SimpleNamespace(named_parameters=self._named_parameters(1.0))
        trainer.rank = 0
        trainer.model_ema = object() if with_ema else None
        trainer.ckpt_store = SimpleNamespace(
            path_for=lambda step: Path(f"model.ckpt-{step}.pt"),
            publish=lambda path: None,
            prune=lambda path: None,
        )
        trainer.ema_ckpt_store = SimpleNamespace(
            path_for=lambda step: Path(f"model.ckpt-ema-{step}.pt"),
            publish=lambda path: None,
            prune=lambda path: None,
        )
        return trainer

    def test_validation_best_raises_before_serialize(self) -> None:
        # Sticky non-finite after a later finite update must still abort
        # validation-best publication before any serialize callback runs.
        trainer = self._bare_trainer()
        trainer.nonfinite_grad_guard.update(torch.tensor(float("nan"), device="cpu"))
        trainer.nonfinite_grad_guard.update(torch.tensor(1.0, device="cpu"))
        writes: list[object] = []

        def fake_write(path, *, step: int, use_ema_weights: bool = False) -> None:
            del step, use_ema_weights
            writes.append(path)

        trainer._save_checkpoint_to_path = fake_write  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            trainer._save_full_validation_checkpoint(Path("best.pt"), step=3)
        self.assertEqual(writes, [])

    def test_ema_validation_best_raises_before_serialize(self) -> None:
        trainer = self._bare_trainer()
        trainer.nonfinite_grad_guard.update(torch.tensor(float("inf"), device="cpu"))
        writes: list[object] = []

        def fake_write(path, *, step: int, use_ema_weights: bool = False) -> None:
            del step
            writes.append((path, use_ema_weights))

        trainer._save_checkpoint_to_path = fake_write  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            trainer._save_full_validation_ema_checkpoint(Path("best-ema.pt"), step=4)
        self.assertEqual(writes, [])

    def test_regular_boundary_validates_once_for_live_and_ema(self) -> None:
        trainer = self._bare_trainer(with_ema=True)
        trainer.nonfinite_grad_guard.update(torch.tensor(1.0, device="cpu"))
        checks = {"count": 0}
        writes: list[bool] = []
        real_ensure = trainer.ensure_finite_gradients_for_checkpoint

        def counting_ensure() -> None:
            checks["count"] += 1
            real_ensure()

        def fake_write(path, *, step: int, use_ema_weights: bool = False) -> None:
            del path, step
            writes.append(use_ema_weights)

        trainer.ensure_finite_gradients_for_checkpoint = (  # type: ignore[method-assign]
            counting_ensure
        )
        trainer._save_checkpoint_to_path = fake_write  # type: ignore[method-assign]

        trainer.save_checkpoint(5)

        self.assertEqual(checks["count"], 1)
        self.assertEqual(writes, [False, True])
        # Successful boundary cleared the flag; a later gate is a no-op.
        trainer.ensure_finite_gradients_for_checkpoint()
        self.assertEqual(checks["count"], 2)


if __name__ == "__main__":
    unittest.main()
