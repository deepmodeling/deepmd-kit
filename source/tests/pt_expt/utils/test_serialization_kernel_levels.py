# SPDX-License-Identifier: LGPL-3.0-or-later
"""Kernel-level selection for pt_expt serialization."""

import os
import sys
from types import (
    ModuleType,
)

import pytest
import torch

from deepmd.pt_expt.utils import (
    serialization,
)

_INFER_LEVELS = (
    "DP_TRITON_INFER",
    "DP_CUDA_INFER",
    "DP_CUTILE_INFER",
    "DP_CUTE_INFER",
)


def _clear_infer_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove inherited accelerator policy from an environment-sensitive test."""
    for name in _INFER_LEVELS:
        monkeypatch.delenv(name, raising=False)


def _capture_pt2_levels(monkeypatch, data, *, lower_kind="nlist"):
    captured = {}

    def capture(*args, **kwargs):
        captured["triton"] = os.environ.get("DP_TRITON_INFER")
        captured["cuda"] = os.environ.get("DP_CUDA_INFER")
        captured["cutile"] = os.environ.get("DP_CUTILE_INFER")
        captured["cute"] = os.environ.get("DP_CUTE_INFER")

    monkeypatch.setattr(serialization, "_deserialize_to_file_pt2", capture)
    serialization.deserialize_to_file("model.pt2", data, lower_kind=lower_kind)
    return captured


def test_dpa4_uses_pt_freeze_defaults_and_restores_environment(monkeypatch) -> None:
    _clear_infer_levels(monkeypatch)

    captured = _capture_pt2_levels(
        monkeypatch,
        {
            "model": {
                "@class": "Model",
                "type": "standard",
                "descriptor": {"@class": "Descriptor", "type": "SeZM"},
            }
        },
    )

    assert captured == {
        "triton": "2",
        "cuda": "1",
        "cutile": "0",
        "cute": "0",
    }
    assert "DP_TRITON_INFER" not in os.environ
    assert "DP_CUDA_INFER" not in os.environ
    assert "DP_CUTILE_INFER" not in os.environ
    assert "DP_CUTE_INFER" not in os.environ


def test_dpa4_explicit_triton_cuda_levels_win(monkeypatch) -> None:
    monkeypatch.setenv("DP_TRITON_INFER", "3")
    monkeypatch.setenv("DP_CUDA_INFER", "0")
    monkeypatch.setenv("DP_CUTILE_INFER", "1")
    monkeypatch.setenv("DP_CUTE_INFER", "1")

    captured = _capture_pt2_levels(
        monkeypatch,
        {"model": {"type": "dpa4"}},
    )

    assert captured == {
        "triton": "3",
        "cuda": "0",
        "cutile": "0",
        "cute": "0",
    }
    assert os.environ["DP_TRITON_INFER"] == "3"
    assert os.environ["DP_CUDA_INFER"] == "0"
    assert os.environ["DP_CUTILE_INFER"] == "1"
    assert os.environ["DP_CUTE_INFER"] == "1"


def test_dpa4_pte_does_not_apply_pt2_kernel_defaults(monkeypatch) -> None:
    _clear_infer_levels(monkeypatch)
    captured = {}

    def capture(*args, **kwargs):
        captured["triton"] = os.environ.get("DP_TRITON_INFER")
        captured["cuda"] = os.environ.get("DP_CUDA_INFER")
        captured["cutile"] = os.environ.get("DP_CUTILE_INFER")
        captured["cute"] = os.environ.get("DP_CUTE_INFER")

    monkeypatch.setattr(serialization, "_deserialize_to_file_pte", capture)
    serialization.deserialize_to_file(
        "model.pte",
        {"model": {"type": "dpa4"}},
        lower_kind="nlist",
    )

    assert captured == {
        "triton": None,
        "cuda": None,
        "cutile": None,
        "cute": None,
    }


def test_dpa4_pte_graph_keeps_legacy_cuda_floor(monkeypatch) -> None:
    _clear_infer_levels(monkeypatch)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    monkeypatch.setattr(
        "deepmd.pt_expt.kernels.utils.backend_device_type", lambda: "cuda"
    )
    captured = {}

    def capture(*args, **kwargs):
        captured["triton"] = os.environ.get("DP_TRITON_INFER")
        captured["cuda"] = os.environ.get("DP_CUDA_INFER")
        captured["cutile"] = os.environ.get("DP_CUTILE_INFER")
        captured["cute"] = os.environ.get("DP_CUTE_INFER")

    monkeypatch.setattr(serialization, "_deserialize_to_file_pte", capture)
    serialization.deserialize_to_file(
        "model.pte",
        {"model": {"type": "dpa4"}},
        lower_kind="graph",
    )

    assert captured == {
        "triton": None,
        "cuda": "2",
        "cutile": None,
        "cute": None,
    }
    assert os.environ["DP_CUDA_INFER"] == "1"


@pytest.mark.parametrize(
    "descriptor_type",
    ["dpa1", "dpa4c"],
)
def test_level_two_graph_families_keep_cuda_floor(monkeypatch, descriptor_type) -> None:
    _clear_infer_levels(monkeypatch)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    monkeypatch.setattr(
        "deepmd.pt_expt.kernels.utils.backend_device_type", lambda: "cuda"
    )

    captured = _capture_pt2_levels(
        monkeypatch,
        {
            "model": {
                "type": "standard",
                "descriptor": {"type": descriptor_type},
            }
        },
        lower_kind="graph",
    )

    assert captured == {
        "triton": None,
        "cuda": "2",
        "cutile": None,
        "cute": None,
    }
    assert os.environ["DP_CUDA_INFER"] == "1"


def test_level_two_graph_family_takes_priority_over_dpa4() -> None:
    assert serialization._uses_dpa4_kernel_defaults(
        {"type": "ener", "descriptor": {"type": "SeZM"}}
    )
    assert not serialization._uses_dpa4_kernel_defaults(
        {"type": "dpa4c", "descriptor": {"type": "dpa4c"}}
    )
    assert not serialization._uses_dpa4_kernel_defaults(
        {
            "type": "hybrid",
            "descriptors": [{"type": "SeZM"}, {"type": "dpa1"}],
        }
    )


@pytest.mark.parametrize(
    ("model_data", "has_value_path", "expected_call"),
    [
        ({"type": "dpa4"}, True, True),
        (
            {
                "type": "standard",
                "descriptor": {"type": "SeZM"},
            },
            True,
            True,
        ),
        ({"type": "dpa1"}, True, False),
        ({"type": "dpa4c"}, True, False),
        (
            {
                "type": "hybrid",
                "descriptors": [{"type": "SeZM"}, {"type": "dpa1"}],
            },
            True,
            False,
        ),
        ({"type": "dpa4"}, False, False),
    ],
)
def test_packed_weights_only_prepare_for_bound_dpa4_value_path(
    monkeypatch,
    model_data: dict,
    has_value_path: bool,
    expected_call: bool,
) -> None:
    model = torch.nn.Module()
    conv = torch.nn.Module()
    conv.triton_infer_l_2_value = object() if has_value_path else None
    model.add_module("conv", conv)

    calls = []
    module_name = "deepmd.pt_expt.kernels.triton.sezm.so2_value_path"
    value_path_module = ModuleType(module_name)
    value_path_module.prepare_triton_value_path_weights = calls.append
    monkeypatch.setitem(sys.modules, module_name, value_path_module)

    serialization._prepare_dpa4_triton_value_path_weights(model, model_data)

    assert calls == ([model] if expected_call else [])


@pytest.mark.parametrize(
    ("model", "target", "expected"),
    [
        ({"type": "dpa4"}, "cpu", ("0", "0", "0", "0")),
        ({"type": "dpa4"}, "cuda", ("1", "1", "0", "0")),
        ({"type": "dpa1"}, "cpu", ("1", "1", "1", "1")),
        ({"type": "dpa4c"}, "cpu", ("1", "1", "1", "1")),
    ],
)
def test_target_policy_only_suppresses_incompatible_dpa4_accelerators(
    monkeypatch,
    model,
    target,
    expected,
) -> None:
    for name in _INFER_LEVELS:
        monkeypatch.setenv(name, "1")

    with serialization._dpa4_kernel_levels_for_target(model, torch.device(target)):
        assert tuple(os.environ[name] for name in _INFER_LEVELS) == expected

    assert all(os.environ[name] == "1" for name in _INFER_LEVELS)
