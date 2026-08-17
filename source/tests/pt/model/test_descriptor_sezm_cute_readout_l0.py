# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused algebra, dispatch, custom-op, and fullgraph readout checks."""

from __future__ import (
    annotations,
)

import importlib

import pytest
import torch

COEFF_DIM = 16
N_FRAMES = 3
PACKED_COEFF_DIM = COEFF_DIM * N_FRAMES
GRID_SIZE = 152
HIDDEN_CHANNELS = 192
PACKED_WIDTH = N_FRAMES * HIDDEN_CHANNELS
TOL = 5.0e-5


def _cute_runtime_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "Neo readout l=0 differentials require CUDA"
    if tuple(torch.cuda.get_device_capability()) not in {(8, 0), (9, 0)}:
        return "Neo readout l=0 differentials require sm80 or sm90"
    try:
        importlib.import_module("cutlass.cute")
        importlib.import_module("cuda.bindings.driver")
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"Neo readout l=0 differentials require CuTe DSL: {exc}"
    return None


_CUTE_SKIP_REASON = _cute_runtime_skip_reason()


def _cpu_inputs(nodes: int = 2, hidden_channels: int = HIDDEN_CHANNELS):
    width = N_FRAMES * hidden_channels
    left = torch.randn(nodes, COEFF_DIM, 1, width, device="cpu")
    right = torch.randn_like(left)
    to_grid = torch.randn(GRID_SIZE, PACKED_COEFF_DIM, device="cpu")
    from_grid = torch.randn(PACKED_COEFF_DIM, GRID_SIZE, device="cpu")
    return left, right, to_grid, from_grid


def _output_ffn(
    *,
    hidden_channels: int = 96,
    trainable: bool = False,
    device: str = "cpu",
):
    from deepmd.pt.model.descriptor.sezm_nn.ffn import (
        EquivariantFFN,
    )

    return (
        EquivariantFFN(
            lmax=3,
            channels=32,
            hidden_channels=hidden_channels,
            kmax=1,
            grid_mlp=True,
            grid_branch=0,
            dtype=torch.float32,
            s2_activation=False,
            ffn_so3_grid=True,
            activation_function="silu",
            glu_activation=True,
            mlp_bias=False,
            trainable=trainable,
            seed=29,
        )
        .to(device)
        .eval()
    )


def _neo_descriptor():
    from deepmd.pt.model.descriptor.sezm import (
        DescrptSeZM,
    )

    return DescrptSeZM(
        ntypes=2,
        sel=4,
        channels=32,
        lmax=3,
        mmax=1,
        n_blocks=2,
        so2_layers=3,
        n_focus=2,
        message_node_so3=True,
        ffn_neurons=0,
        ffn_so3_grid=True,
        grid_branch=[0, 0, 1],
        ffn_blocks=1,
        so3_readout="mlp",
        use_amp=False,
        precision="float32",
        trainable=False,
        seed=42,
    ).eval()


@pytest.fixture
def cpu_neo_descriptor(monkeypatch):
    from deepmd.pt.model.network import (
        mlp,
    )
    from deepmd.pt.utils import (
        env,
    )
    from deepmd.pt.utils import utils as pt_utils

    cpu = torch.device("cpu")
    monkeypatch.setattr(env, "DEVICE", cpu)
    monkeypatch.setattr(pt_utils, "DEVICE", cpu)
    monkeypatch.setattr(mlp, "device", cpu)
    return _neo_descriptor().to("cpu")


def _capture_matmul_precision_state() -> tuple[str | None, str | None, str]:
    matmul = torch.backends.cuda.matmul
    try:
        global_precision = torch.backends.fp32_precision
    except AttributeError:
        return None, None, torch.get_float32_matmul_precision()

    torch.backends.fp32_precision = "none"
    backend_precision = matmul.fp32_precision
    matmul.fp32_precision = "none"
    try:
        legacy_precision = torch.get_float32_matmul_precision()
    finally:
        torch.backends.fp32_precision = global_precision
        matmul.fp32_precision = backend_precision
    return backend_precision, global_precision, legacy_precision


def _restore_matmul_precision_state(
    state: tuple[str | None, str | None, str],
) -> None:
    backend_precision, global_precision, legacy_precision = state
    matmul = torch.backends.cuda.matmul
    if backend_precision is None:
        torch.set_float32_matmul_precision(legacy_precision)
        return
    matmul.fp32_precision = "none"
    if global_precision is not None:
        torch.backends.fp32_precision = "none"
    torch.set_float32_matmul_precision(legacy_precision)
    if global_precision is not None:
        torch.backends.fp32_precision = global_precision
    matmul.fp32_precision = backend_precision


@pytest.fixture
def matmul_precision_state():
    state = _capture_matmul_precision_state()
    try:
        yield torch.backends.cuda.matmul
    finally:
        _restore_matmul_precision_state(state)


def _set_new_matmul_precision(matmul, precision: str) -> None:
    try:
        matmul.fp32_precision = precision
    except AttributeError:
        pytest.skip("new CUDA matmul precision API is unavailable")


def _reference_product(left, right, to_grid, from_grid):
    nodes = left.shape[0]
    left_flat = left.reshape(nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS)
    right_flat = right.reshape(nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS)
    left_grid = torch.einsum("gj,njh->ngh", to_grid, left_flat)
    right_grid = torch.einsum("gj,njh->ngh", to_grid, right_flat)
    return torch.einsum(
        "g,ngh->nh",
        from_grid[0],
        left_grid * right_grid,
    )


def test_sm80_readout_input_fold_selector_uses_only_full_neo_gate(monkeypatch):
    from deepmd.kernels.cute.neo import (
        runtime_policy,
    )

    monkeypatch.delenv("DP_NEO_CUTE_INFER", raising=False)
    monkeypatch.setenv("DP_CUTE_INFER", "1")
    monkeypatch.delenv("DP_CUTE_READOUT_INPUT_FOLD_SM80", raising=False)
    assert not runtime_policy.is_cute_infer_enabled()
    assert not runtime_policy.is_readout_input_fold_sm80_enabled((8, 0))

    monkeypatch.setenv("DP_CUTE_INFER", "0")
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    assert runtime_policy.is_cute_infer_enabled()
    assert runtime_policy.is_readout_input_fold_sm80_enabled((8, 0))
    monkeypatch.setenv("DP_CUTE_READOUT_INPUT_FOLD_SM80", "0")
    assert not runtime_policy.is_readout_input_fold_sm80_enabled((8, 0))
    monkeypatch.setenv("DP_CUTE_READOUT_INPUT_FOLD_SM80", "1")
    assert runtime_policy.is_readout_input_fold_sm80_enabled((8, 0))
    assert not runtime_policy.is_readout_input_fold_sm80_enabled((9, 0))


def test_sm90_readout_input_fold_selector_uses_only_full_neo_gate(monkeypatch):
    from deepmd.kernels.cute.neo import (
        runtime_policy,
    )

    monkeypatch.delenv("DP_NEO_CUTE_INFER", raising=False)
    monkeypatch.setenv("DP_CUTE_INFER", "1")
    monkeypatch.delenv("DP_CUTE_READOUT_INPUT_FOLD_SM90", raising=False)
    assert not runtime_policy.is_readout_input_fold_sm90_enabled((9, 0))

    monkeypatch.setenv("DP_CUTE_INFER", "0")
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    assert runtime_policy.is_readout_input_fold_sm90_enabled((9, 0))
    assert runtime_policy.is_readout_input_fold_enabled((9, 0))
    monkeypatch.setenv("DP_CUTE_READOUT_INPUT_FOLD_SM90", "0")
    assert not runtime_policy.is_readout_input_fold_sm90_enabled((9, 0))
    monkeypatch.setenv("DP_CUTE_READOUT_INPUT_FOLD_SM90", "1")
    assert runtime_policy.is_readout_input_fold_sm90_enabled((9, 0))
    assert not runtime_policy.is_readout_input_fold_sm90_enabled((8, 0))


def test_readout_input_fold_guard_accepts_validated_architectures(monkeypatch):
    from deepmd.kernels.cute.neo import (
        readout_l0,
        runtime_policy,
    )

    module = _output_ffn()
    value = torch.randn(2, COEFF_DIM, 1, 32, device="cpu")
    capabilities = []

    def select_readout(capability):
        capabilities.append(capability)
        return capability in {(8, 0), (9, 0)}

    monkeypatch.setattr(
        runtime_policy,
        "is_readout_input_fold_enabled",
        select_readout,
    )
    monkeypatch.setattr(readout_l0, "_has_exact_neo_readout_contract", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 0))
    assert readout_l0._can_use_sm80_readout_input_fold(module, value)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert readout_l0._can_use_sm80_readout_input_fold(module, value)
    assert capabilities == [(8, 0), (9, 0)]


def test_sm80_readout_input_fold_matches_forward_and_input_vjp(monkeypatch):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    with torch.no_grad():
        module.so3_linear_2.weight.normal_(std=0.03)
    value = torch.randn(3, COEFF_DIM, 1, 32, device="cpu")
    cotangent = torch.randn(3, 32, device="cpu")

    monkeypatch.setattr(
        readout_l0,
        "_can_use_sm80_readout_input_fold",
        lambda *_: False,
    )
    value_ref = value.detach().clone().requires_grad_(True)
    expected = readout_l0._run_neo_readout_l0(
        module,
        value_ref,
        _reference_product,
    )
    expected_grad = torch.autograd.grad(expected, value_ref, cotangent)[0]

    monkeypatch.setattr(
        readout_l0,
        "_can_use_sm80_readout_input_fold",
        lambda *_: True,
    )

    def forbid_staged_projection(*args, **kwargs):
        del args, kwargs
        pytest.fail("folded readout must not execute a staged input projection")

    monkeypatch.setattr(module.so3_linear_1, "forward", forbid_staged_projection)
    monkeypatch.setattr(
        module.act.grid_op.left_proj,
        "forward",
        forbid_staged_projection,
    )
    monkeypatch.setattr(
        module.act.grid_op.right_proj,
        "forward",
        forbid_staged_projection,
    )
    monkeypatch.setattr(module.act.scalar_gate, "forward", forbid_staged_projection)

    def compact_reference(left, right, to_grid, from_grid):
        assert left.is_contiguous()
        assert right.is_contiguous()
        return _reference_product(left, right, to_grid, from_grid)

    value_actual = value.detach().clone().requires_grad_(True)
    actual = readout_l0._run_neo_readout_l0(
        module,
        value_actual,
        compact_reference,
    )
    actual_grad = torch.autograd.grad(actual, value_actual, cotangent)[0]

    torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)
    torch.testing.assert_close(actual_grad, expected_grad, atol=TOL, rtol=TOL)


def test_sm80_readout_input_fold_is_fullgraph_safe(monkeypatch):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    with torch.no_grad():
        module.so3_linear_2.weight.normal_(std=0.03)
    monkeypatch.setattr(
        readout_l0,
        "_can_use_sm80_readout_input_fold",
        lambda *_: True,
    )

    def folded(value):
        return readout_l0._run_neo_readout_l0(
            module,
            value,
            _reference_product,
        )

    value = torch.randn(2, COEFF_DIM, 1, 32, device="cpu")
    readout_l0.prepare_sm80_readout_input_fold(module)
    expected = folded(value)
    compiled = torch.compile(folded, backend="eager", fullgraph=True)
    actual = compiled(value)
    torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)


def test_sm80_readout_input_fold_cache_refreshes_after_inplace_change(monkeypatch):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    ready_calls = []
    monkeypatch.setattr(
        readout_l0,
        "_synchronize_sm80_readout_input_fold_build",
        lambda weights: ready_calls.append(weights),
    )
    first = readout_l0.prepare_sm80_readout_input_fold(module)
    first_cache = getattr(module, readout_l0._READOUT_INPUT_FOLD_CACHE)
    assert getattr(module, readout_l0._READOUT_INPUT_FOLD_LEFT) is first[0]
    assert not any("readout_input_fold" in key for key in module.state_dict())

    cached = readout_l0.prepare_sm80_readout_input_fold(module)
    assert all(
        actual is expected for actual, expected in zip(cached, first, strict=True)
    )
    assert len(ready_calls) == 1
    with torch.no_grad():
        module.act.grid_op.left_proj.weight.add_(0.01)
    second = readout_l0.prepare_sm80_readout_input_fold(module)
    second_cache = getattr(module, readout_l0._READOUT_INPUT_FOLD_CACHE)

    assert second_cache is not first_cache
    assert second[0] is not first[0]
    assert not torch.equal(second[0], first[0])
    assert len(ready_calls) == 2


def test_sm80_readout_input_fold_cache_refreshes_after_parameter_replacement():
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    first = readout_l0.prepare_sm80_readout_input_fold(module)
    old_weight = module.act.grid_op.left_proj.weight
    module.act.grid_op.left_proj.weight = torch.nn.Parameter(
        old_weight.detach().clone().add_(0.02),
        requires_grad=False,
    )
    second = readout_l0.prepare_sm80_readout_input_fold(module)

    assert second[0] is not first[0]
    assert not torch.equal(second[0], first[0])


@pytest.mark.parametrize("assign", [False, True])
def test_sm80_readout_input_fold_load_state_dict_invalidates_cache(assign):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    first = readout_l0.prepare_sm80_readout_input_fold(module)
    state = {key: value.clone() for key, value in module.state_dict().items()}
    left_key = next(key for key in state if key.endswith("grid_op.left_proj.weight"))
    state[left_key].add_(0.03)

    module.load_state_dict(state, assign=assign)

    assert getattr(module, readout_l0._READOUT_INPUT_FOLD_CACHE) is None
    assert all(
        getattr(module, name) is None for name in readout_l0._READOUT_INPUT_FOLD_BUFFERS
    )
    second = readout_l0.prepare_sm80_readout_input_fold(module)
    assert second[0] is not first[0]
    assert not torch.equal(second[0], first[0])
    assert not any("readout_input_fold" in key for key in module.state_dict())


def test_sm80_readout_input_fold_buffers_follow_device_moves():
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    readout_l0.prepare_sm80_readout_input_fold(module)

    module.to("meta")

    assert all(
        getattr(module, name).device.type == "meta"
        for name in readout_l0._READOUT_INPUT_FOLD_BUFFERS
    )
    refreshed = readout_l0.prepare_sm80_readout_input_fold(module)
    assert all(weight.device.type == "meta" for weight in refreshed)


def test_sm80_readout_input_fold_compile_requires_explicit_preparation(
    monkeypatch,
):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="prepare_sm80_readout_input_fold"):
        readout_l0._get_sm80_readout_input_fold(module)


def test_sm80_readout_input_fold_compiled_lookup_returns_prepared_buffers(
    monkeypatch,
):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    expected = readout_l0.prepare_sm80_readout_input_fold(module)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    cached = readout_l0._get_sm80_readout_input_fold(module)
    assert all(
        actual is reference for actual, reference in zip(cached, expected, strict=True)
    )


@pytest.mark.parametrize(
    ("hidden_channels", "expected"),
    [(HIDDEN_CHANNELS, True), (96, False), (128, False)],
)
def test_shape_guard_accepts_only_exact_c192_readout(
    hidden_channels: int,
    expected: bool,
):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    left, _, _, _ = _cpu_inputs(hidden_channels=hidden_channels)
    assert readout_l0._has_exact_product_shape(left) is expected


def test_fake_registrations_return_canonical_strides():
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    shape = (2, COEFF_DIM, 1, PACKED_WIDTH)
    canonical = (COEFF_DIM * PACKED_WIDTH, PACKED_WIDTH, PACKED_WIDTH, 1)
    left = torch.empty_strided(shape, (canonical[0], canonical[1], 7, 1), device="cpu")
    right = torch.empty_strided(
        shape, (canonical[0], canonical[1], 11, 1), device="cpu"
    )
    dq0 = torch.empty(2, HIDDEN_CHANNELS, device="cpu")
    to_grid = torch.empty(GRID_SIZE, PACKED_COEFF_DIM, device="cpu")
    from_grid = torch.empty(PACKED_COEFF_DIM, GRID_SIZE, device="cpu")

    q0 = readout_l0._readout_l0_fake(left, right, to_grid, from_grid)
    grad_left, grad_right = readout_l0._readout_l0_bwd_fake(
        dq0, left, right, to_grid, from_grid
    )

    assert q0.shape == (2, HIDDEN_CHANNELS)
    assert q0.stride() == (HIDDEN_CHANNELS, 1)
    assert grad_left.stride() == canonical
    assert grad_right.stride() == canonical


def test_custom_ops_use_canonical_fake_metadata():
    from torch._subclasses.fake_tensor import (
        FakeTensorMode,
    )

    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    shape = (2, COEFF_DIM, 1, PACKED_WIDTH)
    canonical = (COEFF_DIM * PACKED_WIDTH, PACKED_WIDTH, PACKED_WIDTH, 1)
    with FakeTensorMode():
        left = torch.empty_strided(
            shape, (canonical[0], canonical[1], 7, 1), device="cuda"
        )
        right = torch.empty_strided(
            shape, (canonical[0], canonical[1], 11, 1), device="cuda"
        )
        dq0 = torch.empty(2, HIDDEN_CHANNELS, device="cuda")
        to_grid = torch.empty(GRID_SIZE, PACKED_COEFF_DIM, device="cuda")
        from_grid = torch.empty(PACKED_COEFF_DIM, GRID_SIZE, device="cuda")
        q0 = readout_l0._readout_l0_op(left, right, to_grid, from_grid)
        grad_left, grad_right = readout_l0._readout_l0_bwd_op(
            dq0, left, right, to_grid, from_grid
        )

    assert q0.stride() == (HIDDEN_CHANNELS, 1)
    assert grad_left.stride() == canonical
    assert grad_right.stride() == canonical


def test_reference_completion_matches_module_output_and_input_vjp():
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    with torch.no_grad():
        module.so3_linear_2.weight.normal_(std=0.05)
    value = 0.1 * torch.randn(3, COEFF_DIM, 1, 32, device="cpu")
    seed = torch.randn(3, 32, device="cpu")
    value_ref = value.detach().clone().requires_grad_(True)
    value_actual = value.detach().clone().requires_grad_(True)

    expected = (value_ref + module(value_ref))[:, 0, 0, :]
    expected_grad = torch.autograd.grad(expected, value_ref, seed)[0]
    actual = readout_l0._run_neo_readout_l0(module, value_actual, _reference_product)
    actual_grad = torch.autograd.grad(actual, value_actual, seed)[0]

    torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)
    torch.testing.assert_close(actual_grad, expected_grad, atol=TOL, rtol=TOL)


def test_exact_structure_and_frozen_inference_guards():
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    assert readout_l0._has_exact_neo_readout_structure(module)
    assert readout_l0._inference_mode_is_frozen(module)
    assert not readout_l0._has_exact_neo_readout_structure(
        _output_ffn(hidden_channels=64)
    )
    module.train()
    assert not readout_l0._inference_mode_is_frozen(module)
    module.eval()
    next(module.parameters()).requires_grad_(True)
    assert not readout_l0._inference_mode_is_frozen(module)


def test_module_boundary_falls_back_when_device_is_unsupported(monkeypatch):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    value = torch.randn(2, COEFF_DIM, 1, 32, device="cpu")
    expected = (value + module(value))[:, 0, 0, :]
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")

    assert readout_l0.maybe_run_neo_readout_l0(module, value) is None
    torch.testing.assert_close(
        readout_l0.run_neo_output_readout(module, value), expected
    )


def test_descriptor_freeze_guard_and_readout_wiring(cpu_neo_descriptor, monkeypatch):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    descriptor = cpu_neo_descriptor
    value = torch.randn(2, COEFF_DIM, 1, 32, device="cpu")
    expected = torch.randn(2, 32, device="cpu")
    calls = []

    def record_readout(output_ffn, ffn_in, *, parameters_frozen):
        calls.append((output_ffn, ffn_in, parameters_frozen))
        return expected

    monkeypatch.setattr(readout_l0, "run_neo_output_readout", record_readout)
    assert descriptor._readout_parameters_are_frozen()
    assert descriptor._run_output_readout(value) is expected
    assert calls[-1] == (descriptor.output_ffn, value, True)

    next(descriptor.parameters()).requires_grad_(True)
    descriptor._run_output_readout(value)
    assert calls[-1] == (descriptor.output_ffn, value, False)


def test_trainable_descriptor_state_bypasses_candidate(monkeypatch):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    module = _output_ffn()
    value = torch.randn(2, COEFF_DIM, 1, 32, device="cpu")
    expected = (value + module(value))[:, 0, 0, :]
    monkeypatch.setattr(
        readout_l0,
        "maybe_run_neo_readout_l0",
        lambda *args, **kwargs: pytest.fail("candidate must be bypassed"),
    )

    actual = readout_l0.run_neo_output_readout(module, value, parameters_frozen=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("highest", True), ("high", False)],
)
def test_legacy_matmul_precision_guard_is_eager_and_fullgraph_safe(
    matmul_precision_state,
    precision: str,
    expected: bool,
):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    try:
        matmul_precision_state.fp32_precision = "none"
    except AttributeError:
        pass
    torch.set_float32_matmul_precision(precision)
    value = torch.ones(1, device="cpu")

    def guarded(tensor):
        return tensor if readout_l0._uses_strict_fp32_matmul() else -tensor

    eager = guarded(value)
    fullgraph = torch.compile(guarded, backend="eager", fullgraph=True)(value)
    assert readout_l0._uses_strict_fp32_matmul() is expected
    assert torch.equal(fullgraph, eager)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("ieee", True), ("tf32", False)],
)
def test_modern_matmul_precision_guard_rejects_tf32(
    matmul_precision_state,
    precision: str,
    expected: bool,
):
    from deepmd.kernels.cute.neo import (
        readout_l0,
    )

    _set_new_matmul_precision(matmul_precision_state, precision)
    assert readout_l0._uses_strict_fp32_matmul() is expected


@pytest.mark.skipif(
    _CUTE_SKIP_REASON is not None,
    reason=_CUTE_SKIP_REASON or "CuTe runtime unavailable",
)
class TestReadoutL0Cuda:
    @staticmethod
    def _inputs(nodes: int):
        generator = torch.Generator(device="cuda").manual_seed(20260704 + nodes)
        left = 0.1 * torch.randn(
            nodes,
            COEFF_DIM,
            1,
            PACKED_WIDTH,
            device="cuda",
            generator=generator,
        )
        right = 0.1 * torch.randn(left.shape, device="cuda", generator=generator)
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

    @pytest.mark.parametrize("nodes", [1, 7, 65])
    def test_forward_and_input_vjp_match_strict_fp32(
        self,
        nodes: int,
    ):
        from deepmd.kernels.cute.neo.readout_l0 import (
            readout_l0_product_cute,
        )

        left, right, to_grid, from_grid = self._inputs(nodes)
        left_ref = left.detach().clone().requires_grad_(True)
        right_ref = right.detach().clone().requires_grad_(True)
        left_actual = left.detach().clone().requires_grad_(True)
        right_actual = right.detach().clone().requires_grad_(True)
        dq0 = torch.randn(nodes, HIDDEN_CHANNELS, device="cuda")

        expected = _reference_product(left_ref, right_ref, to_grid, from_grid)
        expected_grads = torch.autograd.grad(expected, (left_ref, right_ref), dq0)
        actual = readout_l0_product_cute(left_actual, right_actual, to_grid, from_grid)
        actual_grads = torch.autograd.grad(actual, (left_actual, right_actual), dq0)

        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)
        torch.testing.assert_close(
            actual_grads[0], expected_grads[0], atol=TOL, rtol=TOL
        )
        torch.testing.assert_close(
            actual_grads[1], expected_grads[1], atol=TOL, rtol=TOL
        )

    def test_opcheck_and_fullgraph_preserve_metadata_and_vjp(self):
        from deepmd.kernels.cute.neo import (
            readout_l0,
        )

        left, right, to_grid, from_grid = self._inputs(3)
        dq0 = torch.randn(3, HIDDEN_CHANNELS, device="cuda")
        torch.library.opcheck(
            readout_l0._readout_l0_op,
            (left, right, to_grid, from_grid),
            test_utils=("test_schema", "test_faketensor"),
        )
        compiled = torch.compile(
            readout_l0.readout_l0_product_cute,
            dynamic=True,
            fullgraph=True,
        )
        left.requires_grad_(True)
        right.requires_grad_(True)
        actual = compiled(left, right, to_grid, from_grid)
        grads = torch.autograd.grad(actual, (left, right), dq0)
        assert actual.shape == (3, HIDDEN_CHANNELS)
        assert grads[0].shape == left.shape
        assert grads[1].shape == right.shape

    def test_exact_module_fullgraph_uses_full_neo_gate(self, monkeypatch):
        from deepmd.kernels.cute.neo import (
            readout_l0,
        )

        module = _output_ffn(device="cuda")
        with torch.no_grad():
            module.so3_linear_2.weight.normal_(std=0.05)
        value = torch.randn(3, COEFF_DIM, 1, 32, device="cuda")
        seed = torch.randn(3, 32, device="cuda")
        value_ref = value.detach().clone().requires_grad_(True)
        value_actual = value.detach().clone().requires_grad_(True)
        monkeypatch.delenv("DP_NEO_CUTE_INFER", raising=False)
        expected = (value_ref + module(value_ref))[:, 0, 0, :]
        expected_grad = torch.autograd.grad(expected, value_ref, seed)[0]

        monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
        readout_l0.prepare_sm80_readout_input_fold(module)
        compiled = torch.compile(
            lambda tensor: readout_l0.run_neo_output_readout(module, tensor),
            dynamic=True,
            fullgraph=True,
        )
        actual = compiled(value_actual)
        actual_grad = torch.autograd.grad(actual, value_actual, seed)[0]

        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)
        torch.testing.assert_close(actual_grad, expected_grad, atol=TOL, rtol=TOL)

    def test_sm80_input_fold_preparation_is_cross_stream_ready(self):
        from deepmd.kernels.cute.neo import (
            readout_l0,
        )

        if tuple(torch.cuda.get_device_capability()) not in {(8, 0), (8, 6)}:
            pytest.skip("readout input folding requires SM80 or SM86")

        module = _output_ffn(device="cuda")
        expected = readout_l0._build_sm80_readout_input_fold(module)
        torch.cuda.synchronize()

        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            prepared = readout_l0.prepare_sm80_readout_input_fold(module)
        assert producer.query()

        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            observed = tuple(weight.clone() for weight in prepared)
        consumer.synchronize()
        for actual, reference in zip(observed, expected, strict=True):
            torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)

    def test_sm80_input_fold_fullgraph_matches_module_vjp(self, monkeypatch):
        from deepmd.kernels.cute.neo import (
            readout_l0,
        )

        if tuple(torch.cuda.get_device_capability()) not in {(8, 0), (8, 6)}:
            pytest.skip("readout input folding requires SM80 or SM86")

        module = _output_ffn(device="cuda")
        with torch.no_grad():
            module.so3_linear_2.weight.normal_(std=0.05)
        value = torch.randn(7, COEFF_DIM, 1, 32, device="cuda")
        seed = torch.randn(7, 32, device="cuda")
        value_ref = value.detach().clone().requires_grad_(True)
        value_actual = value.detach().clone().requires_grad_(True)
        expected = (value_ref + module(value_ref))[:, 0, 0, :]
        expected_grad = torch.autograd.grad(expected, value_ref, seed)[0]

        monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
        monkeypatch.setenv("DP_CUTE_READOUT_INPUT_FOLD_SM80", "1")
        assert readout_l0._can_use_sm80_readout_input_fold(module, value_actual)
        readout_l0.prepare_sm80_readout_input_fold(module)
        compiled = torch.compile(
            lambda tensor: readout_l0.run_neo_output_readout(module, tensor),
            dynamic=True,
            fullgraph=True,
        )
        actual = compiled(value_actual)
        actual_grad = torch.autograd.grad(actual, value_actual, seed)[0]

        torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)
        torch.testing.assert_close(actual_grad, expected_grad, atol=TOL, rtol=TOL)
