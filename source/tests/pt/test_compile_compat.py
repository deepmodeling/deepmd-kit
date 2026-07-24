# SPDX-License-Identifier: LGPL-3.0-or-later
"""forbidden_dims_from_model accessor handling (CodeRabbit #5779)."""

import torch

from deepmd.pt.utils.compile_compat import (
    forbidden_dims_from_model,
)


class _WithDims(torch.nn.Module):
    def get_dim_fparam(self) -> int:
        return 5

    def get_dim_aparam(self) -> int:
        return 3


class TestForbiddenDimsFromModel:
    def test_dims_collected_when_accessors_present(self) -> None:
        forbidden = forbidden_dims_from_model(_WithDims(), [])
        assert {3, 5} <= forbidden

    def test_missing_accessors_fall_through_best_effort(self) -> None:
        # a bare Module lacks get_dim_fparam/get_dim_aparam: the lookup must
        # happen inside the try (an eagerly-built accessor tuple raised
        # AttributeError before the best-effort guard could catch it)
        assert forbidden_dims_from_model(torch.nn.Module(), []) == set()
