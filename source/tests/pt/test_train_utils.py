# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import torch

from deepmd.pt.train.utils import (
    clip_grad_norm_with_stable_fallback,
)


class TestStableGradClip(unittest.TestCase):
    def test_fallback_clips_large_finite_gradients(self) -> None:
        p0, p1 = self._make_large_grad_parameters()

        with patch(
            "torch.nn.utils.clip_grad_norm_",
            side_effect=RuntimeError("non-finite total norm"),
        ):
            total_norm = clip_grad_norm_with_stable_fallback(
                [p0, p1],
                max_norm=3.0,
                named_parameters=lambda: [("p0", p0), ("p1", p1)],
            )

        self._check_clipped_norm(total_norm, p0, p1)

    def test_real_overflow_path_uses_stable_fallback(self) -> None:
        p0, p1 = self._make_large_grad_parameters()

        total_norm = clip_grad_norm_with_stable_fallback(
            [p0, p1],
            max_norm=3.0,
            named_parameters=lambda: [("p0", p0), ("p1", p1)],
        )

        self._check_clipped_norm(total_norm, p0, p1)

    def _make_large_grad_parameters(
        self,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        p0 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))
        p1 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))
        p0.grad = torch.tensor([torch.finfo(torch.float32).max / 2], device="cpu")
        p1.grad = torch.tensor([torch.finfo(torch.float32).max / 2], device="cpu")
        return p0, p1

    def _check_clipped_norm(
        self,
        total_norm: torch.Tensor,
        p0: torch.nn.Parameter,
        p1: torch.nn.Parameter,
    ) -> None:
        clipped_norm = torch.linalg.vector_norm(
            torch.stack([p0.grad.double().norm(), p1.grad.double().norm()])
        )
        self.assertTrue(torch.isfinite(total_norm))
        self.assertEqual(total_norm.dtype, torch.float64)
        self.assertAlmostEqual(clipped_norm.item(), 3.0, places=5)
