# SPDX-License-Identifier: LGPL-3.0-or-later
"""forbidden_dims_from_model accessor handling (CodeRabbit #5779)."""

import pytest
import torch

from deepmd.pt.utils import (
    compile_compat,
)
from deepmd.pt.utils.compile_compat import (
    build_inductor_compile_options,
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


def test_fusion_size_defaults_to_eight(monkeypatch) -> None:
    monkeypatch.delenv("DP_FUSION_SIZE", raising=False)
    monkeypatch.delenv("DP_GEN_UNDER_SANITIZER", raising=False)
    monkeypatch.setattr(compile_compat, "_torch_release", lambda: (2, 13))

    assert build_inductor_compile_options()["max_fusion_size"] == 8
    inference_options = build_inductor_compile_options(inference=True)
    assert inference_options["max_fusion_size"] == 8
    assert inference_options["cpp.min_chunk_size"] == 1
    assert inference_options["cpp.dynamic_threads"] is True


def test_torch_211_inference_keeps_default_cpp_parallelism(monkeypatch) -> None:
    monkeypatch.delenv("DP_GEN_UNDER_SANITIZER", raising=False)
    monkeypatch.setattr(compile_compat, "_torch_release", lambda: (2, 11))

    inference_options = build_inductor_compile_options(inference=True)

    assert "cpp.min_chunk_size" not in inference_options
    assert "cpp.dynamic_threads" not in inference_options


def test_fusion_size_environment_is_shared(monkeypatch) -> None:
    monkeypatch.setenv("DP_FUSION_SIZE", "16")

    assert build_inductor_compile_options()["max_fusion_size"] == 16
    assert build_inductor_compile_options(inference=True)["max_fusion_size"] == 16


def test_lsan_inference_uses_serial_codegen(monkeypatch) -> None:
    monkeypatch.setenv("DP_GEN_UNDER_SANITIZER", "lsan")

    training_options = build_inductor_compile_options()
    inference_options = build_inductor_compile_options(inference=True)

    assert "cpp.threads" not in training_options
    assert "cpp.dynamic_threads" not in training_options
    assert "cpp.min_chunk_size" not in inference_options
    assert inference_options["cpp.dynamic_threads"] is False
    assert inference_options["cpp.threads"] == 1


@pytest.mark.parametrize("value", ["0", "-1", "fast"])
def test_fusion_size_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("DP_FUSION_SIZE", value)

    with pytest.raises(ValueError, match="DP_FUSION_SIZE must be a positive integer"):
        build_inductor_compile_options()
