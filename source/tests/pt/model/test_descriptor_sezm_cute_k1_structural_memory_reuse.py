# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Behavioral contracts for K1 structural storage reuse."""

from __future__ import (
    annotations,
)

import importlib.util
import sys
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)

from .test_descriptor_sezm_cute_k1 import (
    _K1,
    NeoFullCuteBackward,
    NeoK1BackwardWorkspace,
    NeoK1RuntimeConfig,
    StackCache,
    _validate_runtime_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
STRUCTURAL_HELPER = REPO_ROOT / "deepmd/pt_expt/kernels/cute/sezm/k1_gate_structural.py"
SO2_HELPER = REPO_ROOT / "deepmd/pt_expt/kernels/cute/sezm/k1_so2linear.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _valid_structural_config() -> NeoK1RuntimeConfig:
    return NeoK1RuntimeConfig(per_focus_so2_fwd_pair=True)


def _storage_ptr(tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _workspace_inputs(edge_count: int):
    torch = _K1.torch
    return {
        "edge_count": edge_count,
        "node_count": 3,
        "d_full": torch.empty(edge_count, 46, dtype=torch.float32, device="cpu"),
        "dt_full": torch.empty(edge_count, 46, dtype=torch.float32, device="cpu"),
        "radial": torch.empty(edge_count, 4, 32, dtype=torch.float32, device="cpu"),
    }


def _so2_linear(torch):
    return SimpleNamespace(
        lmax=3,
        mmax=1,
        in_channels=32,
        out_channels=32,
        n_focus=2,
        mlp_bias=False,
        weight_m0=torch.randn(4 * 32, 2 * 4 * 32, device="cpu"),
        weight_m=(torch.randn(3 * 32, 2 * 2 * 3 * 32, device="cpu"),),
    )


def test_structural_memory_reuse_profile_is_valid() -> None:
    config = _valid_structural_config()
    assert _validate_runtime_config(config, compute_capability=(8, 0)) is None


def test_final_so2_single_input_fold_matches_residual_plus_linear():
    torch = _K1.torch
    helper = _load_module("sezm_cute_k1_so2_test", SO2_HELPER)
    torch.manual_seed(20260705)
    x_local = torch.randn(3, 2, 10, 32, dtype=torch.float32, device="cpu")
    linear = _so2_linear(torch)

    linear_only = helper.run_neo_so2_linear_manual(linear, x_local)
    folded = helper.run_neo_so2_linear_manual(
        linear,
        x_local,
        add_residual=True,
    )

    torch.testing.assert_close(folded, linear_only + x_local, atol=5e-5, rtol=5e-5)


def test_per_focus_pair_forward_matches_batched_forward():
    torch = _K1.torch
    helper = _load_module("sezm_cute_k1_so2_per_focus_test", SO2_HELPER)
    torch.manual_seed(20260718)
    x_local = torch.randn(17, 2, 10, 32, dtype=torch.float32, device="cpu")
    linear = _so2_linear(torch)

    batched = helper.run_neo_so2_linear_manual(linear, x_local)
    per_focus = helper.run_neo_so2_linear_manual(
        linear,
        x_local,
        per_focus_pair=True,
    )

    torch.testing.assert_close(per_focus, batched, atol=5e-5, rtol=5e-5)


def test_structural_workspace_reuses_saved_slabs_and_gate_panel():
    torch = _K1.torch
    edge_count = 5
    phase_c_stack = torch.empty(
        edge_count, 2, 10, 32, dtype=torch.float32, device="cpu"
    )
    phase_c_y = torch.empty_like(phase_c_stack)
    radial_scratch = torch.empty(
        2, edge_count, 3 * 32, dtype=torch.float32, device="cpu"
    )

    workspace = NeoK1BackwardWorkspace(
        torch,
        **_workspace_inputs(edge_count),
        phase_c_stack=phase_c_stack,
        phase_c_y=phase_c_y,
        radial_scratch=radial_scratch,
        structural_memory_reuse=True,
    )

    stack_storage = _storage_ptr(phase_c_stack)
    assert _storage_ptr(workspace.grad_stack_focus) == stack_storage
    assert _storage_ptr(workspace.grad_y) == stack_storage
    assert _storage_ptr(workspace.grad_x_rot) == stack_storage
    assert _storage_ptr(workspace.grad_d) == stack_storage
    assert _storage_ptr(workspace.grad_radial_flat) == _storage_ptr(radial_scratch)
    assert _storage_ptr(workspace.grad_mixed_slab) == _storage_ptr(phase_c_y)
    assert workspace.grad_gate_logits is None


def test_single_input_workspace_omits_phase_c_y_and_keeps_mixed_output_distinct():
    torch = _K1.torch
    edge_count = 5
    phase_c_stack = torch.empty(
        edge_count, 2, 10, 32, dtype=torch.float32, device="cpu"
    )
    radial_scratch = torch.empty(edge_count, 4, 32, dtype=torch.float32, device="cpu")

    workspace = NeoK1BackwardWorkspace(
        torch,
        **_workspace_inputs(edge_count),
        phase_c_stack=phase_c_stack,
        phase_c_y=None,
        radial_scratch=radial_scratch,
        structural_memory_reuse=True,
        phase_c_single_input_reuse=True,
    )

    assert _storage_ptr(workspace.grad_stack_focus) == _storage_ptr(phase_c_stack)
    assert _storage_ptr(workspace.grad_mixed_slab) != _storage_ptr(phase_c_stack)
    assert _storage_ptr(workspace.grad_radial_flat) == _storage_ptr(radial_scratch)
    assert workspace.grad_mixed_slab.shape == phase_c_stack.shape


def test_runner_routes_saved_structural_buffers_into_lazy_workspace():
    torch = _K1.torch
    edge_count = 5
    runner = object.__new__(NeoFullCuteBackward)
    runner.torch = torch
    runner.config = _valid_structural_config()
    runner.edge_count = edge_count
    runner.node_count = 3
    runner.d = torch.empty(edge_count, 46, dtype=torch.float32, device="cpu")
    runner.dt = torch.empty_like(runner.d)
    runner.radial = torch.empty(edge_count, 4, 32, dtype=torch.float32, device="cpu")
    runner.phase_c_stack = torch.empty(
        edge_count, 2, 10, 32, dtype=torch.float32, device="cpu"
    )
    runner.phase_c_y = None
    first_logits = torch.empty(2, edge_count, 3 * 32, dtype=torch.float32, device="cpu")
    runner.stack_caches = [
        StackCache(
            torch.empty_like(runner.phase_c_stack), first_logits, object(), False
        ),
        StackCache(
            torch.empty_like(runner.phase_c_stack),
            torch.empty_like(first_logits),
            object(),
            False,
        ),
        StackCache(torch.empty_like(runner.phase_c_stack), None, object(), True),
    ]
    runner._backward_workspace = None

    workspace = runner.ensure_backward_workspace()

    assert runner.ensure_backward_workspace() is workspace
    assert _storage_ptr(workspace.grad_stack_focus) == _storage_ptr(
        runner.phase_c_stack
    )
    assert _storage_ptr(workspace.grad_radial_flat) == _storage_ptr(first_logits)
    assert _storage_ptr(workspace.grad_mixed_slab) != _storage_ptr(runner.phase_c_stack)


def test_structural_backward_can_overwrite_consumed_logits():
    torch = _K1.torch
    helper = _load_module("sezm_cute_k1_structural_test", STRUCTURAL_HELPER)
    edge_count = 3
    grad_out = torch.randn(edge_count, 2, 10, 32, dtype=torch.float32, device="cpu")
    y = torch.randn_like(grad_out)
    logits = torch.randn(2, edge_count, 3 * 32, dtype=torch.float32, device="cpu")
    grad_y = torch.empty_like(y)

    def fake_kernel(grad_out_flat, y_flat, logits_in, grad_y_flat, grad_logits):
        assert logits_in is logits
        assert grad_logits is logits
        grad_y_flat.copy_(grad_out_flat + y_flat)
        grad_logits.fill_(7.0)

    result = helper.run_structural_gate_backward(
        fake_kernel,
        grad_out,
        y,
        logits,
        grad_y,
        grad_logits=None,
        overwrite_logits=True,
    )

    assert result is logits
    torch.testing.assert_close(grad_y, grad_out + y)
