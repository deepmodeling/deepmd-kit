# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU contracts for the batched Neo radial adjoint projection."""

from __future__ import (
    annotations,
)

import unittest
from types import (
    SimpleNamespace,
)
from unittest import (
    mock,
)

import torch

from deepmd.kernels.cute.neo import k1_radial_phase_a_node as _RADIAL_PROJECTION
from deepmd.kernels.cute.neo import k1_runner as _K1_RUNNER

COMPACT_WIDTH = _RADIAL_PROJECTION.COMPACT_WIDTH
FOCUS_COUNT = _RADIAL_PROJECTION.FOCUS_COUNT
FOCUS_HIDDEN = _RADIAL_PROJECTION.FOCUS_HIDDEN
PROJECTION_INPUT_WIDTH = _RADIAL_PROJECTION.PROJECTION_INPUT_WIDTH
RADIAL_WIDTH = _RADIAL_PROJECTION.RADIAL_WIDTH
_project_batched_radial_adjoint = _RADIAL_PROJECTION._project_batched_radial_adjoint
prepare_batched_radial_projection_weight = (
    _RADIAL_PROJECTION.prepare_batched_radial_projection_weight
)
_batched_radial_projection_weight = _K1_RUNNER._batched_radial_projection_weight


class TestBatchedRadialProjection(unittest.TestCase):
    def test_precombined_single_gemm_matches_two_projection_algebra(self):
        torch.manual_seed(20260718)
        edge_count = 7
        grad_compact = torch.randn(
            edge_count, COMPACT_WIDTH, dtype=torch.float32, device="cpu"
        )
        grad_logits = torch.randn(
            edge_count, FOCUS_COUNT, dtype=torch.float32, device="cpu"
        )
        combined_weight = torch.randn(
            RADIAL_WIDTH,
            COMPACT_WIDTH,
            dtype=torch.float32,
            device="cpu",
        )
        attention_weight = torch.randn(
            FOCUS_HIDDEN,
            FOCUS_COUNT,
            dtype=torch.float32,
            device="cpu",
        )
        projection_weight = prepare_batched_radial_projection_weight(
            combined_weight,
            attention_weight,
        )

        expected = grad_compact @ combined_weight.transpose(0, 1)
        expected[:, :FOCUS_HIDDEN].add_(grad_logits @ attention_weight.transpose(0, 1))
        workspace = torch.full(
            (edge_count, FOCUS_COUNT * 10 * FOCUS_HIDDEN),
            torch.nan,
            dtype=torch.float32,
            device="cpu",
        )
        actual = torch.empty(
            edge_count,
            RADIAL_WIDTH,
            dtype=torch.float32,
            device="cpu",
        )
        workspace[:, :COMPACT_WIDTH].copy_(grad_compact)
        workspace[:, COMPACT_WIDTH:PROJECTION_INPUT_WIDTH].copy_(grad_logits)

        with (
            mock.patch.object(torch, "cat", wraps=torch.cat) as cat,
            mock.patch.object(torch, "mm", wraps=torch.mm) as mm,
        ):
            _project_batched_radial_adjoint(
                projection_weight,
                actual,
                workspace,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        self.assertEqual(mm.call_count, 1)
        self.assertEqual(cat.call_count, 0)
        gemm_input, gemm_weight = mm.call_args.args
        self.assertEqual(tuple(gemm_input.shape), (edge_count, PROJECTION_INPUT_WIDTH))
        self.assertEqual(
            tuple(gemm_weight.shape), (PROJECTION_INPUT_WIDTH, RADIAL_WIDTH)
        )
        self.assertEqual(gemm_input.stride(), (workspace.shape[1], 1))
        self.assertEqual(
            gemm_input.untyped_storage().data_ptr(),
            workspace.untyped_storage().data_ptr(),
        )
        self.assertIs(mm.call_args.kwargs["out"], actual)
        torch.testing.assert_close(
            workspace[:, :COMPACT_WIDTH],
            grad_compact,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            workspace[:, COMPACT_WIDTH:PROJECTION_INPUT_WIDTH],
            grad_logits,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.isnan(workspace[:, PROJECTION_INPUT_WIDTH:]).all())

    def test_precombined_weight_layout_and_cache_contract(self):
        combined_weight = torch.randn(
            RADIAL_WIDTH,
            COMPACT_WIDTH,
            dtype=torch.float32,
            device="cpu",
        )
        attention_weight = torch.randn(
            FOCUS_HIDDEN,
            FOCUS_COUNT,
            dtype=torch.float32,
            device="cpu",
        )
        owner = SimpleNamespace()

        first = _batched_radial_projection_weight(
            owner,
            combined_weight,
            attention_weight,
        )
        second = _batched_radial_projection_weight(
            owner,
            combined_weight,
            attention_weight,
        )

        self.assertIs(first, second)
        self.assertEqual(tuple(first.shape), (PROJECTION_INPUT_WIDTH, RADIAL_WIDTH))
        self.assertTrue(first.is_contiguous())
        torch.testing.assert_close(
            first[:COMPACT_WIDTH],
            combined_weight.transpose(0, 1),
            rtol=0.0,
            atol=0.0,
        )
        attention_panel = torch.zeros(
            FOCUS_COUNT,
            2 * FOCUS_HIDDEN,
            device="cpu",
        )
        attention_panel[:, :FOCUS_HIDDEN].copy_(attention_weight.transpose(0, 1))
        torch.testing.assert_close(
            first[COMPACT_WIDTH:, : 2 * FOCUS_HIDDEN],
            attention_panel,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.count_nonzero(first[COMPACT_WIDTH:, 64:]) == 0)

        attention_weight.add_(1.0)
        updated = _batched_radial_projection_weight(
            owner,
            combined_weight,
            attention_weight,
        )
        self.assertIsNot(first, updated)

    def test_projection_requires_strict_fp32_and_full_consumed_workspace(self):
        combined_weight = torch.randn(
            RADIAL_WIDTH,
            COMPACT_WIDTH,
            device="cpu",
        )
        attention_weight = torch.randn(
            FOCUS_HIDDEN,
            FOCUS_COUNT,
            device="cpu",
        )
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            prepare_batched_radial_projection_weight(
                combined_weight.double(),
                attention_weight.double(),
            )

        projection_weight = prepare_batched_radial_projection_weight(
            combined_weight,
            attention_weight,
        )
        edge_count = 2
        with self.assertRaisesRegex(ValueError, "consumed_workspace must have shape"):
            _project_batched_radial_adjoint(
                projection_weight,
                torch.empty(edge_count, RADIAL_WIDTH, device="cpu"),
                torch.empty(edge_count, PROJECTION_INPUT_WIDTH, device="cpu"),
            )


if __name__ == "__main__":
    unittest.main()
