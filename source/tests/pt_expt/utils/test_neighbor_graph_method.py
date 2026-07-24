# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for resolve_auto_graph_builder availability ladder."""

from unittest.mock import (
    patch,
)

import pytest
import torch

from deepmd.pt_expt.utils.neighbor_graph_method import (
    resolve_auto_graph_builder,
)


@pytest.mark.parametrize(
    "nv,vesin,expected",
    [
        (True, True, "nv"),
        (True, False, "nv"),
        (False, True, "vesin"),
        (False, False, "dense"),
    ],
)
def test_cuda_ladder(nv: bool, vesin: bool, expected: str) -> None:
    with (
        patch(
            "deepmd.pt_expt.utils.neighbor_graph_method.is_nv_available",
            return_value=nv,
        ),
        patch(
            "deepmd.pt_expt.utils.neighbor_graph_method.is_vesin_torch_available",
            return_value=vesin,
        ),
    ):
        assert resolve_auto_graph_builder(torch.device("cuda")) == expected
        assert resolve_auto_graph_builder("cuda:0") == expected


@pytest.mark.parametrize(
    "vesin,expected",
    [
        (True, "vesin"),
        (False, "dense"),
    ],
)
def test_cpu_ladder(vesin: bool, expected: str) -> None:
    with (
        patch(
            "deepmd.pt_expt.utils.neighbor_graph_method.is_nv_available",
            return_value=True,  # ignored on CPU
        ),
        patch(
            "deepmd.pt_expt.utils.neighbor_graph_method.is_vesin_torch_available",
            return_value=vesin,
        ),
    ):
        assert resolve_auto_graph_builder(torch.device("cpu")) == expected
        assert resolve_auto_graph_builder("cpu") == expected
