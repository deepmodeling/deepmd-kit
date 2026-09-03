# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""SM80/SM90 differential coverage for Phase-C focus-source load fusion."""

from __future__ import (
    annotations,
)

import unittest

import pytest
import torch


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in {
        (8, 0),
        (9, 0),
    }


def test_source_csr_preserves_equal_source_order() -> None:
    from deepmd.pt_expt.kernels.cute.sezm.k1_radial_phase_a_node import (
        build_source_csr,
    )

    src = torch.tensor([2, 0, 2, 1, 0, 2], dtype=torch.int64, device="cpu")
    source_csr = build_source_csr(src, node_count=3)

    torch.testing.assert_close(
        source_csr.source_order,
        torch.tensor([1, 4, 3, 0, 2, 5], dtype=torch.int32, device="cpu"),
    )


@pytest.mark.parametrize("src_values", ([-1, 0], [0, 3]))
def test_source_csr_rejects_out_of_range_sources(src_values: list[int]) -> None:
    from deepmd.pt_expt.kernels.cute.sezm.k1_radial_phase_a_node import (
        build_source_csr,
    )

    src = torch.tensor(src_values, dtype=torch.int64, device="cpu")
    with pytest.raises(RuntimeError, match="source indices"):
        build_source_csr(src, node_count=3)


def test_source_csr_eager_validation_reports_value_error() -> None:
    from deepmd.pt_expt.kernels.cute.sezm.k1_radial_phase_a_node import (
        build_source_csr,
    )

    src = torch.tensor([0, 3], dtype=torch.int64, device="cpu")
    with pytest.raises(ValueError, match="source indices"):
        build_source_csr(src, node_count=3, validate_sources=True)


@unittest.skipUnless(_has_supported_gpu(), "requires an SM80 or SM90 CUDA device")
class TestRadialFocusSourceFusion(unittest.TestCase):
    def test_fused_load_matches_standalone_scalar_lane_add(self) -> None:
        from deepmd.pt_expt.kernels.cute.sezm.k1_radial_phase_a_node import (
            build_source_csr,
            prepare_batched_radial_projection_weight,
            run_neo_radial_phase_a_backward_node_tiled,
        )

        torch.manual_seed(18072026)
        device = torch.device("cuda")
        edge_count = 7
        node_count = 3

        src = torch.tensor([0, 2, 1, 0, 2, 1, 2], device=device, dtype=torch.int64)
        source_csr = build_source_csr(src, node_count)
        grad_stack = torch.randn(edge_count, 2, 10, 32, device=device)
        grad_focus_src = torch.randn(2, edge_count, 32, device=device)
        grad_logits = torch.randn(edge_count, 2, device=device)
        radial_compact = torch.randn(edge_count, 25, device=device)
        combined_weight = torch.randn(128, 25, device=device)
        attention_weight = torch.randn(32, 2, device=device)
        channel_basis = torch.randn(64, device=device)
        x_wide = torch.randn(node_count, 16 * 64, device=device)
        d_full = torch.randn(edge_count, 46, device=device)
        projection_weight = prepare_batched_radial_projection_weight(
            combined_weight,
            attention_weight,
        )

        standalone_grad_stack = grad_stack.clone()
        standalone_grad_stack[:, :, 0, :].add_(grad_focus_src.permute(1, 0, 2))
        standalone = run_neo_radial_phase_a_backward_node_tiled(
            standalone_grad_stack.view(edge_count, 2 * 10 * 32),
            grad_logits,
            radial_compact,
            channel_basis,
            x_wide,
            source_csr.source_order,
            source_csr.source_ptr,
            d_full,
            grad_focus_src_focus=torch.zeros_like(grad_focus_src),
            batched_radial_projection_weight=projection_weight,
        )
        fused = run_neo_radial_phase_a_backward_node_tiled(
            grad_stack.view(edge_count, 2 * 10 * 32),
            grad_logits,
            radial_compact,
            channel_basis,
            x_wide,
            source_csr.source_order,
            source_csr.source_ptr,
            d_full,
            grad_focus_src_focus=grad_focus_src,
            batched_radial_projection_weight=projection_weight,
        )

        torch.testing.assert_close(
            fused.grad_x_wide,
            standalone.grad_x_wide,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            fused.grad_d_full,
            standalone.grad_d_full,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            fused.grad_radial_m0,
            standalone.grad_radial_m0,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
