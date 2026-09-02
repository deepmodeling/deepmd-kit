# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Strict-FP32 differentials for the tiled Neo output-grid product."""

from __future__ import (
    annotations,
)

import ast
import importlib
from pathlib import (
    Path,
)

import pytest
import torch

TOL = 5.0e-5
PACKED_COEFF_DIM = 48
GRID_SIZE = 152
SUPPORTED_HIDDEN_CHANNELS = (96, 192)
REPO_ROOT = Path(__file__).resolve().parents[4]
TILED_KERNEL_PATH = (
    REPO_ROOT
    / "deepmd/pt_expt/kernels/cute/sezm/output_grid_kernels"
    / "cute_tiled_grid_product.py"
)
MESSAGE_GRID_PATH = (
    REPO_ROOT
    / "deepmd/pt_expt/kernels/cute/sezm/k1_kernels"
    / "cute_neo_message_grid_product.py"
)


def _cute_runtime_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "tiled output-grid differentials require CUDA"
    if torch.cuda.get_device_capability()[0] < 8:
        return "tiled output-grid differentials require compute capability 8.0+"
    try:
        importlib.import_module("cutlass.cute")
        importlib.import_module("cuda.bindings.driver")
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"tiled output-grid differentials require CuTe DSL: {exc}"
    return None


_CUTE_SKIP_REASON = _cute_runtime_skip_reason()


def _sm80_skip_reason() -> str | None:
    if _CUTE_SKIP_REASON is not None:
        return _CUTE_SKIP_REASON
    if tuple(torch.cuda.get_device_capability()) not in {(8, 0), (8, 6)}:
        return "specialized output-grid differentials require sm80 or sm86"
    return None


_SM80_SKIP_REASON = _sm80_skip_reason()


def _method_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_sm80_c96_n48_panel_tiles_shared_input_before_partition_b() -> None:
    """Guard the CuTe B-fragment layout contract without requiring CUDA."""
    tree = ast.parse(TILED_KERNEL_PATH.read_text(encoding="utf-8"))
    method = _method_node(tree, "_backproject_panel_accumulate")

    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    shared_b = assignments["sB"]
    assert isinstance(shared_b, ast.Call)
    assert ast.unparse(shared_b.func) == "cute.local_tile"
    assert ast.unparse(shared_b.args[0]) == "panel_b"
    keywords = {
        keyword.arg: ast.unparse(keyword.value) for keyword in shared_b.keywords
    }
    assert keywords == {
        "tiler": "self.cta_tiler",
        "coord": "(0, 0, None)",
        "proj": "(None, 1, 1)",
    }

    partition_b = assignments["tSsB"]
    assert isinstance(partition_b, ast.Call)
    assert ast.unparse(partition_b.func) == "thr_mma.partition_B"
    assert ast.unparse(partition_b.args[0]) == "sB"
    assert "thr_mma.partition_B(panel_b)" not in ast.unparse(method)


def test_sm80_c96_n48_panel_retains_one_ordered_accumulator() -> None:
    """Guard the strict-FP32 K-tile order of the panel decomposition."""
    tree = ast.parse(TILED_KERNEL_PATH.read_text(encoding="utf-8"))
    parent = _method_node(tree, "_panel_adjoint_backward")
    helper = _method_node(tree, "_backproject_panel_accumulate")
    parent_source = ast.unparse(parent)
    helper_source = ast.unparse(helper)

    assert "for grid_tile in cutlass.range_constexpr(GRID_TILES)" in parent_source
    assert parent_source.count("self._backproject_panel_accumulate(") == 2
    assert "tCrOut_left.fill(0.0)" in parent_source
    assert "tCrOut_right.fill(0.0)" in parent_source
    assert ".fill(0.0)" not in helper_source
    assert "panel_k_start = grid_tile * (TILE_M // self.tile_k)" in helper_source
    assert "logical_k_tile = cutlass.Int32(0)" in helper_source
    assert "logical_k_tile = logical_k_tile + 1" in helper_source


def test_packed_message_grid_initializes_the_panel_adjoint_selector() -> None:
    """Keep the manually constructed backward operation compile-complete."""
    tree = ast.parse(MESSAGE_GRID_PATH.read_text(encoding="utf-8"))
    factory = _method_node(tree, "_make_grid_operation")
    assigned_attributes = {
        target.attr
        for node in ast.walk(factory)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "operation"
    }
    assert "sm80_c96_n48_panel" in assigned_attributes
    assert "channel_tile_start" in assigned_attributes
    assert "panel_adjoint" not in assigned_attributes


def _reference(left, right, to_grid, from_grid):
    left_grid = torch.einsum("gj,njc->ngc", to_grid, left)
    right_grid = torch.einsum("gj,njc->ngc", to_grid, right)
    return torch.einsum("jg,ngc->njc", from_grid, left_grid * right_grid)


def _reference_backward(grad_out, left, right, to_grid, from_grid):
    grad_product = torch.einsum("jg,njc->ngc", from_grid, grad_out)
    left_grid = torch.einsum("gj,njc->ngc", to_grid, left)
    right_grid = torch.einsum("gj,njc->ngc", to_grid, right)
    grad_left = torch.einsum("gj,ngc->njc", to_grid, grad_product * right_grid)
    grad_right = torch.einsum("gj,ngc->njc", to_grid, grad_product * left_grid)
    return grad_left, grad_right


def _inputs(nodes: int, hidden_channels: int):
    generator = torch.Generator(device="cuda").manual_seed(
        20260703 + nodes + hidden_channels
    )
    left = 0.1 * torch.randn(
        nodes,
        PACKED_COEFF_DIM,
        hidden_channels,
        device="cuda",
        generator=generator,
    )
    right = 0.1 * torch.randn(
        left.shape,
        device="cuda",
        generator=generator,
    )
    to_grid = 0.1 * torch.randn(
        GRID_SIZE,
        PACKED_COEFF_DIM,
        device="cuda",
        generator=generator,
    )
    from_grid = 0.1 * torch.randn(
        PACKED_COEFF_DIM,
        GRID_SIZE,
        device="cuda",
        generator=generator,
    )
    return left, right, to_grid, from_grid


def test_architecture_policy_shares_sm80_backend_with_sm86():
    from deepmd.pt_expt.kernels.cute.sezm.runtime_policy import (
        PORTABLE_TILED_BACKEND,
        PYTORCH_BACKEND,
        output_grid_arch_key,
        select_output_grid_backend,
    )

    for hidden_channels in SUPPORTED_HIDDEN_CHANNELS:
        assert select_output_grid_backend((8, 0), hidden_channels) == (
            PORTABLE_TILED_BACKEND
        )
        assert select_output_grid_backend((8, 6), hidden_channels) == (
            PORTABLE_TILED_BACKEND
        )
        assert select_output_grid_backend((9, 0), hidden_channels) == (
            PORTABLE_TILED_BACKEND
        )
        for capability in ((7, 5), (8, 9), (10, 0), (12, 0)):
            assert (
                select_output_grid_backend(capability, hidden_channels)
                == PYTORCH_BACKEND
            )
    assert select_output_grid_backend((8, 0), 128) == PYTORCH_BACKEND
    assert select_output_grid_backend((8, 6), 128) == PYTORCH_BACKEND
    assert select_output_grid_backend((9, 0), 128) == PYTORCH_BACKEND
    assert output_grid_arch_key((8, 0)) == "sm80"
    assert output_grid_arch_key((8, 6)) == "sm80"
    assert output_grid_arch_key((9, 0)) == "sm90"


@pytest.mark.skipif(
    _CUTE_SKIP_REASON is not None,
    reason=_CUTE_SKIP_REASON or "CuTe runtime unavailable",
)
class TestTiledOutputGridProductForwardCuda:
    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    @pytest.mark.parametrize("nodes", [1, 7, 65])
    def test_matches_strict_fp32_reference(
        self,
        nodes: int,
        hidden_channels: int,
    ):
        from deepmd.pt_expt.kernels.cute.sezm.output_grid_kernels.cute_tiled_grid_product import (
            run_tiled_output_grid_product,
        )

        left, right, to_grid, from_grid = _inputs(nodes, hidden_channels)
        expected = _reference(left, right, to_grid, from_grid)
        actual = run_tiled_output_grid_product(left, right, to_grid, from_grid)
        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)

    @pytest.mark.skipif(
        _SM80_SKIP_REASON is not None,
        reason=_SM80_SKIP_REASON or "sm80-family GPU is unavailable",
    )
    @pytest.mark.parametrize("nodes", [1, 7, 65])
    def test_sm80_c96_n48_matches_strict_fp32_reference(self, nodes: int):
        from deepmd.pt_expt.kernels.cute.sezm.output_grid_kernels.cute_tiled_grid_product import (
            run_tiled_output_grid_product,
        )

        left, right, to_grid, from_grid = _inputs(nodes, 96)
        expected = _reference(left, right, to_grid, from_grid)
        actual = run_tiled_output_grid_product(
            left,
            right,
            to_grid,
            from_grid,
            use_sm80_c96_n48=True,
        )
        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)

    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    def test_one_compile_accepts_symbolic_node_counts(self, hidden_channels: int):
        from deepmd.pt_expt.kernels.cute.sezm.output_grid_kernels.cute_tiled_grid_product import (
            _compiled_tiled_forward,
            run_tiled_output_grid_product,
        )

        before = _compiled_tiled_forward.cache_info()
        for nodes in (3, 19):
            left, right, to_grid, from_grid = _inputs(nodes, hidden_channels)
            actual = run_tiled_output_grid_product(
                left,
                right,
                to_grid,
                from_grid,
            )
            torch.testing.assert_close(
                actual,
                _reference(left, right, to_grid, from_grid),
                atol=TOL,
                rtol=TOL,
            )
        after = _compiled_tiled_forward.cache_info()
        assert after.misses - before.misses <= 1
        assert after.hits > before.hits

    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    def test_launches_on_current_non_default_stream(self, hidden_channels: int):
        from deepmd.pt_expt.kernels.cute.sezm.output_grid_kernels.cute_tiled_grid_product import (
            run_tiled_output_grid_product,
        )

        left, right, to_grid, from_grid = _inputs(11, hidden_channels)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = run_tiled_output_grid_product(
                left,
                right,
                to_grid,
                from_grid,
            )
            expected = _reference(left, right, to_grid, from_grid)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)


@pytest.mark.skipif(
    _CUTE_SKIP_REASON is not None,
    reason=_CUTE_SKIP_REASON or "CuTe runtime unavailable",
)
class TestTiledOutputGridProductBackwardCuda:
    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    @pytest.mark.parametrize("nodes", [1, 7, 65])
    def test_first_backward_matches_strict_fp32_reference(
        self,
        nodes: int,
        hidden_channels: int,
    ):
        from deepmd.pt_expt.kernels.cute.sezm.output_grid_kernels.cute_tiled_grid_product import (
            run_tiled_output_grid_product_backward,
        )

        left, right, to_grid, from_grid = _inputs(nodes, hidden_channels)
        grad_out = torch.randn_like(left)
        expected = _reference_backward(grad_out, left, right, to_grid, from_grid)
        actual = run_tiled_output_grid_product_backward(
            grad_out,
            left,
            right,
            to_grid,
            from_grid,
        )
        torch.testing.assert_close(actual[0], expected[0], atol=TOL, rtol=TOL)
        torch.testing.assert_close(actual[1], expected[1], atol=TOL, rtol=TOL)

    @pytest.mark.skipif(
        _SM80_SKIP_REASON is not None,
        reason=_SM80_SKIP_REASON or "sm80-family GPU is unavailable",
    )
    @pytest.mark.parametrize("nodes", [1, 7, 65])
    def test_sm80_c96_n48_panel_matches_strict_fp32_reference(
        self,
        nodes: int,
    ):
        from deepmd.pt_expt.kernels.cute.sezm.output_grid_kernels.cute_tiled_grid_product import (
            run_tiled_output_grid_product_backward,
        )

        left, right, to_grid, from_grid = _inputs(nodes, 96)
        grad_out = torch.randn_like(left)
        expected = _reference_backward(grad_out, left, right, to_grid, from_grid)
        actual = run_tiled_output_grid_product_backward(
            grad_out,
            left,
            right,
            to_grid,
            from_grid,
            use_sm80_c96_n48_panel=True,
        )
        torch.testing.assert_close(actual[0], expected[0], atol=TOL, rtol=TOL)
        torch.testing.assert_close(actual[1], expected[1], atol=TOL, rtol=TOL)

    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    def test_one_backward_compile_accepts_symbolic_node_counts(
        self,
        hidden_channels: int,
    ):
        from deepmd.pt_expt.kernels.cute.sezm.output_grid_kernels.cute_tiled_grid_product import (
            _compiled_tiled_backward,
            run_tiled_output_grid_product_backward,
        )

        before = _compiled_tiled_backward.cache_info()
        for nodes in (3, 19):
            left, right, to_grid, from_grid = _inputs(nodes, hidden_channels)
            grad_out = torch.randn_like(left)
            expected = _reference_backward(grad_out, left, right, to_grid, from_grid)
            actual = run_tiled_output_grid_product_backward(
                grad_out,
                left,
                right,
                to_grid,
                from_grid,
            )
            torch.testing.assert_close(actual[0], expected[0], atol=TOL, rtol=TOL)
            torch.testing.assert_close(actual[1], expected[1], atol=TOL, rtol=TOL)
        after = _compiled_tiled_backward.cache_info()
        assert after.misses - before.misses <= 1
        assert after.hits > before.hits
