# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Behavioral differentials for the fused Neo output-grid product."""

from __future__ import (
    annotations,
)

import importlib
import sys
import types

import pytest
import torch

TOL = 5.0e-5
N_FRAMES = 3
COEFF_DIM = 16
GRID_SIZE = 152
SUPPORTED_HIDDEN_CHANNELS = (96, 192)


def _cute_runtime_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "output-grid differentials require CUDA"
    if tuple(torch.cuda.get_device_capability()) not in {(8, 0), (8, 6), (9, 0)}:
        return "output-grid CuTe dispatch supports only sm80, sm86, and sm90"
    try:
        importlib.import_module("cutlass.cute")
        importlib.import_module("cuda.bindings.driver")
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"output-grid differentials require the CuTe DSL runtime: {exc}"
    return None


_CUTE_SKIP_REASON = _cute_runtime_skip_reason()


def _sm80_skip_reason() -> str | None:
    if _CUTE_SKIP_REASON is not None:
        return _CUTE_SKIP_REASON
    if tuple(torch.cuda.get_device_capability()) not in {(8, 0), (8, 6)}:
        return "specialized output-grid differentials require sm80 or sm86"
    return None


_SM80_SKIP_REASON = _sm80_skip_reason()


def _reference(left, right, to_grid, from_grid):
    nodes = left.shape[0]
    hidden_channels = left.shape[-1] // N_FRAMES
    left_flat = left.reshape(nodes, COEFF_DIM * N_FRAMES, hidden_channels)
    right_flat = right.reshape(nodes, COEFF_DIM * N_FRAMES, hidden_channels)
    left_grid = torch.einsum("gj,njc->ngc", to_grid, left_flat)
    right_grid = torch.einsum("gj,njc->ngc", to_grid, right_flat)
    out = torch.einsum("jg,ngc->njc", from_grid, left_grid * right_grid)
    return out.reshape_as(left)


def test_grid_mlp_accepts_fused_middle_callback():
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        GridMLP,
    )

    module = GridMLP(
        channels=2,
        mode="self",
        n_frames=3,
        dtype=torch.float32,
        trainable=False,
        seed=7,
    ).to("cpu")
    left = torch.randn(2, 4, 1, 6, device="cpu")
    right = torch.randn_like(left)
    scalar_pair = torch.empty(2, 1, 4, device="cpu")
    calls = 0

    def fused_middle(projected_left, projected_right):
        nonlocal calls
        calls += 1
        return projected_left * projected_right

    out = module(
        left,
        right,
        scalar_pair,
        to_grid=lambda value: value,
        from_grid=lambda value: value,
        grid_product=fused_middle,
    )

    assert calls == 1
    assert out.shape == left.shape


@pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
def test_dispatch_falls_back_without_exact_cuda_contract(
    monkeypatch: pytest.MonkeyPatch,
    hidden_channels: int,
):
    from deepmd.kernels.cute.neo import (
        output_grid_product,
    )

    monkeypatch.setattr(
        output_grid_product.runtime_policy,
        "is_cute_infer_enabled",
        lambda: True,
    )
    left = torch.randn(2, COEFF_DIM, 1, N_FRAMES * hidden_channels, device="cpu")
    right = torch.randn_like(left)
    to_grid = torch.randn(GRID_SIZE, COEFF_DIM * N_FRAMES, device="cpu")
    from_grid = torch.randn(COEFF_DIM * N_FRAMES, GRID_SIZE, device="cpu")

    assert (
        output_grid_product.maybe_run_cute_output_grid_product(
            left,
            right,
            to_grid,
            from_grid,
            n_frames=N_FRAMES,
        )
        is None
    )


@pytest.mark.parametrize(
    ("hidden_channels", "expected"),
    [(96, 96), (192, 192), (128, None)],
)
def test_exact_shape_guard_accepts_only_validated_widths(
    hidden_channels: int,
    expected: int | None,
) -> None:
    from deepmd.kernels.cute.neo.output_grid_product import (
        _exact_hidden_channels,
    )

    left = torch.empty(2, COEFF_DIM, 1, N_FRAMES * hidden_channels, device="cpu")
    assert _exact_hidden_channels(left, N_FRAMES) == expected


@pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
def test_fake_registrations_allocate_canonical_strides(hidden_channels: int) -> None:
    from deepmd.kernels.cute.neo import (
        output_grid_product,
    )

    width = N_FRAMES * hidden_channels
    shape = (2, COEFF_DIM, 1, width)
    canonical_stride = (COEFF_DIM * width, width, width, 1)
    left = torch.empty_strided(
        shape,
        (canonical_stride[0], canonical_stride[1], 7, 1),
        device="cpu",
    )
    right = torch.empty_strided(
        shape,
        (canonical_stride[0], canonical_stride[1], 11, 1),
        device="cpu",
    )
    grad_out = torch.empty_strided(
        shape,
        (canonical_stride[0], canonical_stride[1], 13, 1),
        device="cpu",
    )
    to_grid = torch.empty(GRID_SIZE, COEFF_DIM * N_FRAMES, device="cpu")
    from_grid = torch.empty(COEFF_DIM * N_FRAMES, GRID_SIZE, device="cpu")

    out = output_grid_product._output_grid_product_fake(
        left,
        right,
        to_grid,
        from_grid,
        N_FRAMES,
    )
    grad_left, grad_right = output_grid_product._output_grid_product_bwd_fake(
        grad_out,
        left,
        right,
        to_grid,
        from_grid,
        N_FRAMES,
    )

    assert out.stride() == canonical_stride
    assert grad_left.stride() == canonical_stride
    assert grad_right.stride() == canonical_stride


@pytest.mark.parametrize(
    ("compute_capability", "hidden_channels", "policy_enabled", "expected"),
    [
        ((9, 0), 96, True, True),
        ((9, 0), 96, False, False),
        ((9, 0), 192, True, False),
        ((9, 1), 96, True, False),
    ],
)
def test_sm90_c96_asymmetric_panel_dispatch_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    compute_capability: tuple[int, int],
    hidden_channels: int,
    policy_enabled: bool,
    expected: bool,
) -> None:
    from deepmd.kernels.cute.neo import (
        output_grid_product,
    )

    kernel_module_name = (
        "deepmd.kernels.cute.neo.output_grid_kernels.cute_tiled_grid_product"
    )
    kernel_module = types.ModuleType(kernel_module_name)
    calls: dict[str, dict[str, bool]] = {}

    def run_forward(left, right, to_grid, from_grid, **kwargs):
        del right, to_grid, from_grid
        calls["forward"] = kwargs
        return torch.empty_like(left)

    def run_backward(grad_out, left, right, to_grid, from_grid, **kwargs):
        del grad_out, to_grid, from_grid
        calls["backward"] = kwargs
        return torch.empty_like(left), torch.empty_like(right)

    kernel_module.run_tiled_output_grid_product = run_forward
    kernel_module.run_tiled_output_grid_product_backward = run_backward
    monkeypatch.setitem(sys.modules, kernel_module_name, kernel_module)
    monkeypatch.setattr(
        output_grid_product,
        "_validate_exact_contract",
        lambda *_args: hidden_channels,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device=None: compute_capability,
    )
    monkeypatch.setattr(
        output_grid_product.runtime_policy,
        "is_output_grid_fwd_sm80_c96_n48_enabled",
        lambda _compute_capability: False,
    )
    monkeypatch.setattr(
        output_grid_product.runtime_policy,
        "is_output_grid_bwd_sm80_c96_n48_panel_enabled",
        lambda _compute_capability: False,
    )
    monkeypatch.setattr(
        output_grid_product.runtime_policy,
        "is_output_grid_sm90_c96_asymmetric_panels_enabled",
        lambda _compute_capability: policy_enabled,
    )

    width = N_FRAMES * hidden_channels
    left = torch.empty(2, COEFF_DIM, 1, width, device="cpu")
    right = torch.empty_like(left)
    grad_out = torch.empty_like(left)
    to_grid = torch.empty(
        GRID_SIZE,
        COEFF_DIM * N_FRAMES,
        device="cpu",
    )
    from_grid = torch.empty(
        COEFF_DIM * N_FRAMES,
        GRID_SIZE,
        device="cpu",
    )

    output_grid_product._output_grid_product_impl(
        left,
        right,
        to_grid,
        from_grid,
        N_FRAMES,
    )
    output_grid_product._output_grid_product_bwd_impl(
        grad_out,
        left,
        right,
        to_grid,
        from_grid,
        N_FRAMES,
    )

    assert calls["forward"] == {
        "use_sm80_c96_n48": False,
        "use_sm90_c96_asymmetric_panels": expected,
    }
    assert calls["backward"] == {
        "use_sm80_c96_n48_panel": False,
        "use_sm90_c96_asymmetric_panels": expected,
    }


@pytest.mark.skipif(
    _CUTE_SKIP_REASON is not None,
    reason=_CUTE_SKIP_REASON or "CuTe runtime unavailable",
)
class TestOutputGridProductCuda:
    @staticmethod
    def _inputs(nodes: int, hidden_channels: int):
        generator = torch.Generator(device="cuda").manual_seed(
            20260703 + nodes + hidden_channels
        )
        left = 0.1 * torch.randn(
            nodes,
            COEFF_DIM,
            1,
            N_FRAMES * hidden_channels,
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
            COEFF_DIM * N_FRAMES,
            device="cuda",
            generator=generator,
        )
        from_grid = 0.1 * torch.randn(
            COEFF_DIM * N_FRAMES,
            GRID_SIZE,
            device="cuda",
            generator=generator,
        )
        return left, right, to_grid, from_grid

    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    @pytest.mark.parametrize("nodes", [1, 7, 65])
    def test_forward_and_first_backward_match_strict_fp32(
        self,
        nodes: int,
        hidden_channels: int,
    ):
        from deepmd.kernels.cute.neo.output_grid_product import (
            output_grid_product_cute,
        )

        left, right, to_grid, from_grid = self._inputs(nodes, hidden_channels)
        left_ref = left.detach().clone().requires_grad_(True)
        right_ref = right.detach().clone().requires_grad_(True)
        left_actual = left.detach().clone().requires_grad_(True)
        right_actual = right.detach().clone().requires_grad_(True)
        grad = torch.randn_like(left)

        expected = _reference(left_ref, right_ref, to_grid, from_grid)
        expected_grads = torch.autograd.grad(
            expected,
            (left_ref, right_ref),
            grad,
        )
        actual = output_grid_product_cute(
            left_actual,
            right_actual,
            to_grid,
            from_grid,
            n_frames=N_FRAMES,
        )
        actual_grads = torch.autograd.grad(
            actual,
            (left_actual, right_actual),
            grad,
        )

        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)
        torch.testing.assert_close(
            actual_grads[0], expected_grads[0], atol=TOL, rtol=TOL
        )
        torch.testing.assert_close(
            actual_grads[1], expected_grads[1], atol=TOL, rtol=TOL
        )

    @pytest.mark.skipif(
        _SM80_SKIP_REASON is not None,
        reason=_SM80_SKIP_REASON or "sm80-family GPU is unavailable",
    )
    @pytest.mark.parametrize("nodes", [1, 7, 65])
    def test_sm80_c96_n48_forward_custom_op_matches_strict_fp32(
        self,
        monkeypatch: pytest.MonkeyPatch,
        nodes: int,
    ):
        from deepmd.kernels.cute.neo import (
            output_grid_product,
        )

        monkeypatch.setattr(
            output_grid_product.runtime_policy,
            "is_output_grid_fwd_sm80_c96_n48_enabled",
            lambda _compute_capability: True,
        )
        left, right, to_grid, from_grid = self._inputs(nodes, 96)
        expected = _reference(left, right, to_grid, from_grid)
        actual = output_grid_product.output_grid_product_cute(
            left,
            right,
            to_grid,
            from_grid,
            n_frames=N_FRAMES,
        )
        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)

    @pytest.mark.skipif(
        _SM80_SKIP_REASON is not None,
        reason=_SM80_SKIP_REASON or "sm80-family GPU is unavailable",
    )
    def test_sm80_c96_n48_panel_custom_op_matches_strict_fp32(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from deepmd.kernels.cute.neo import (
            output_grid_product,
        )

        monkeypatch.setattr(
            output_grid_product.runtime_policy,
            "is_output_grid_bwd_sm80_c96_n48_panel_enabled",
            lambda _compute_capability: True,
        )
        left, right, to_grid, from_grid = self._inputs(7, 96)
        left_ref = left.detach().clone().requires_grad_(True)
        right_ref = right.detach().clone().requires_grad_(True)
        left_actual = left.detach().clone().requires_grad_(True)
        right_actual = right.detach().clone().requires_grad_(True)
        grad = torch.randn_like(left)

        expected = _reference(left_ref, right_ref, to_grid, from_grid)
        expected_grads = torch.autograd.grad(
            expected,
            (left_ref, right_ref),
            grad,
        )
        actual = output_grid_product.output_grid_product_cute(
            left_actual,
            right_actual,
            to_grid,
            from_grid,
            n_frames=N_FRAMES,
        )
        actual_grads = torch.autograd.grad(
            actual,
            (left_actual, right_actual),
            grad,
        )

        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)
        torch.testing.assert_close(
            actual_grads[0], expected_grads[0], atol=TOL, rtol=TOL
        )
        torch.testing.assert_close(
            actual_grads[1], expected_grads[1], atol=TOL, rtol=TOL
        )

    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    def test_dispatch_uses_only_the_master_gate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hidden_channels: int,
    ):
        from deepmd.kernels.cute.neo import (
            output_grid_product,
        )

        left, right, to_grid, from_grid = self._inputs(2, hidden_channels)
        monkeypatch.setattr(
            output_grid_product.runtime_policy,
            "is_cute_infer_enabled",
            lambda: False,
        )
        assert (
            output_grid_product.maybe_run_cute_output_grid_product(
                left,
                right,
                to_grid,
                from_grid,
                n_frames=N_FRAMES,
            )
            is None
        )

        monkeypatch.setattr(
            output_grid_product.runtime_policy,
            "is_cute_infer_enabled",
            lambda: True,
        )
        actual = output_grid_product.maybe_run_cute_output_grid_product(
            left,
            right,
            to_grid,
            from_grid,
            n_frames=N_FRAMES,
        )
        assert actual is not None
        torch.testing.assert_close(
            actual,
            _reference(left, right, to_grid, from_grid),
            atol=TOL,
            rtol=TOL,
        )

    @pytest.mark.parametrize("hidden_channels", SUPPORTED_HIDDEN_CHANNELS)
    def test_dispatch_declines_non_strict_matmul_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hidden_channels: int,
    ):
        from deepmd.kernels.cute.neo import (
            output_grid_product,
        )

        left, right, to_grid, from_grid = self._inputs(2, hidden_channels)
        monkeypatch.setattr(
            output_grid_product.runtime_policy,
            "is_cute_infer_enabled",
            lambda: True,
        )
        monkeypatch.setattr(
            output_grid_product.runtime_policy,
            "uses_strict_fp32_matmul",
            lambda: False,
        )

        assert (
            output_grid_product.maybe_run_cute_output_grid_product(
                left,
                right,
                to_grid,
                from_grid,
                n_frames=N_FRAMES,
            )
            is None
        )
