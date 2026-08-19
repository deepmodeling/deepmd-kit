# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numerical differentials for the structural Neo K1 gate."""

from __future__ import (
    annotations,
)

import unittest

import torch

from deepmd.kernels.cute.neo import (
    k1_gate_structural,
)


class TestStructuralGateHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = k1_gate_structural

    def setUp(self):
        torch.manual_seed(20260703)
        self.edge_count = 7
        self.gate_src = torch.randn(
            self.edge_count, 2, 32, dtype=torch.float32, device="cpu"
        )
        self.weight = torch.randn(32, 2 * 3 * 32, dtype=torch.float32, device="cpu")

    def test_focus_major_forward_matches_focus_linear(self):
        actual = self.helper.focus_major_gate_linear_forward(
            self.gate_src,
            self.weight,
        )
        expected = torch.einsum(
            "efi,ifo->efo",
            self.gate_src,
            self.weight.view(32, 2, 3 * 32),
        )
        self.assertEqual(actual.shape, (2, self.edge_count, 3 * 32))
        self.assertEqual(actual.stride(), (self.edge_count * 3 * 32, 3 * 32, 1))
        torch.testing.assert_close(
            actual.permute(1, 0, 2), expected, atol=5e-5, rtol=5e-5
        )

    def test_backward_addmm_accumulates_without_replacing_grad_y_storage(self):
        grad_y = torch.randn(
            self.edge_count, 2, 10, 32, dtype=torch.float32, device="cpu"
        )
        grad_logits = torch.randn(
            2, self.edge_count, 3 * 32, dtype=torch.float32, device="cpu"
        )
        expected = grad_y.clone()
        weight = self.weight.view(32, 2, 3 * 32)
        for focus in range(2):
            expected[:, focus, 0, :] += grad_logits[focus] @ weight[:, focus, :].T

        pointer = grad_y.untyped_storage().data_ptr()
        actual = self.helper.focus_major_gate_linear_backward_add_(
            grad_y,
            grad_logits,
            self.weight,
        )

        self.assertIs(actual, grad_y)
        self.assertEqual(actual.untyped_storage().data_ptr(), pointer)
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)

    def test_forward_wrapper_preserves_caller_owned_storage(self):
        residual = torch.randn(
            self.edge_count, 2, 10, 32, dtype=torch.float32, device="cpu"
        )
        y = torch.randn_like(residual)
        logits = torch.randn(
            2, self.edge_count, 3 * 32, dtype=torch.float32, device="cpu"
        )
        pointer = residual.untyped_storage().data_ptr()

        def fake_kernel(residual_flat, y_flat, logits_arg, out_flat):
            self.assertEqual(residual_flat.data_ptr(), out_flat.data_ptr())
            self.assertIs(logits_arg, logits)
            out_flat.copy_(residual_flat + y_flat)

        actual = self.helper.run_structural_gate_forward(
            fake_kernel,
            residual,
            y,
            logits,
            out=residual,
        )

        self.assertIs(actual, residual)
        self.assertEqual(actual.untyped_storage().data_ptr(), pointer)


if __name__ == "__main__":
    unittest.main()
