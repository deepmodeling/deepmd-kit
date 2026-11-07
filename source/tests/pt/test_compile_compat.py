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
    assert "cpp.max_horizontal_fusion_size" not in inference_options


def test_torch_211_inference_keeps_default_cpp_parallelism(monkeypatch) -> None:
    monkeypatch.delenv("DP_GEN_UNDER_SANITIZER", raising=False)
    monkeypatch.setattr(compile_compat, "_torch_release", lambda: (2, 11))

    inference_options = build_inductor_compile_options(inference=True)

    assert "cpp.min_chunk_size" not in inference_options
    assert "cpp.dynamic_threads" not in inference_options
    assert inference_options["cpp.max_horizontal_fusion_size"] == 1
    assert "cpp.simdlen" not in inference_options
    assert "cpp.max_horizontal_fusion_size" not in build_inductor_compile_options()


def test_inductor_kernels_write_separate_output_buffers() -> None:
    """Generated kernels write separate output buffers instead of an input."""

    def fn(
        a: torch.Tensor, b: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The product is an extern kernel. Where in-place buffers are allowed,
        # the fused kernel of the addition and the weighted sum stores the
        # shifted values over the buffer of the product.
        shifted = a @ b + 1.0
        return shifted, (shifted * w).sum(-1)

    from torch._inductor.utils import (
        run_and_get_code,
    )

    a, b, w = (
        torch.randn(shape, dtype=torch.float64, device="cpu")
        for shape in ((16, 9), (9, 9), (16, 9))
    )
    expected = a @ b + 1.0

    def generated_code(options: dict) -> list[str]:
        compiled = torch.compile(fn, backend="inductor", options=options)
        (shifted, total), codes = run_and_get_code(compiled, a, b, w)
        torch.testing.assert_close(shifted, expected)
        torch.testing.assert_close(total, (expected * w).sum(-1))
        return codes

    options = build_inductor_compile_options()
    # The same graph with in-place buffers allowed must store over its input,
    # so that the assertion below tests the option rather than the graph.
    in_place = generated_code({**options, "inplace_buffers": True})
    separate = generated_code(options)
    torch._dynamo.reset()

    assert any("in_out_ptr" in code for code in in_place)
    assert not any("in_out_ptr" in code for code in separate)


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
