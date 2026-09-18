# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persistent-complex strict-FP32 Neo SO2 stack for SM90.

The fixed Neo ``m=1`` block is a complex 96-wide representation of the dense
real block ``[[U,V],[-V,U]]``. The first two frozen SO2Linear layers and gated
residuals remain in this representation without intermediate packing.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)
from torch import (
    Tensor,
)

from ...compile_cache import (
    device_aware_lru_cache,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN202, TC002

FOCUS_COUNT = 2
CHANNELS = 32
M0_ROWS = 4
M1_ROWS = 3
M0_WIDTH = M0_ROWS * CHANNELS
M1_WIDTH = M1_ROWS * CHANNELS
PAIR_WIDTH = 2 * M1_WIDTH
GATE_GROUPS = 3
GATE_WIDTH = GATE_GROUPS * CHANNELS
GATED_LAYERS = 2
STACK_LAYERS = 3

ROWS_PER_BLOCK = 8
THREADS = ROWS_PER_BLOCK * CHANNELS
WEIGHT_SMEM_STRIDE = GATE_WIDTH + 1
WEIGHT_VALUES = CHANNELS * GATE_WIDTH
WEIGHT_LOADS_PER_THREAD = WEIGHT_VALUES // THREADS
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}

__all__ = [
    "NeoPersistentComplexSaved",
    "NeoPersistentComplexState",
    "NeoPersistentComplexWeights",
    "prepare_neo_persistent_complex_weights",
    "validate_neo_persistent_complex_state",
]


@dataclass(frozen=True)
class NeoPersistentComplexState:
    """Focus-major state consumed directly by the persistent stack.

    ``m0`` has shape ``(2,E,128)`` and dtype ``float32``.  ``m1`` has shape
    ``(2,E,96)`` and dtype ``complex64``.  Both tensors are contiguous; no
    packing or transposition is needed by the real or complex batched GEMMs.
    """

    m0: Tensor
    m1: Tensor

    @property
    def edge_count(self) -> int:
        return int(self.m0.shape[1])

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in (self.m0, self.m1)
        )


@dataclass(frozen=True)
class NeoPersistentComplexWeights:
    """Frozen strict-FP32 operands in forward and input-adjoint orientation."""

    w0: Tensor
    wc: Tensor
    w0_h: Tensor
    wc_h: Tensor
    gate: Tensor


@dataclass(frozen=True)
class NeoPersistentComplexSaved:
    """Minimal exact gate state for the two nonlinear layers."""

    z0: tuple[Tensor, Tensor]
    z1: tuple[Tensor, Tensor]

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in (*self.z0, *self.z1)
        )


def _require_shape(name: str, tensor: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _require_frozen_cuda_tensor(
    name: str,
    tensor: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.requires_grad:
        raise ValueError(f"{name} must be frozen; parameter gradients are out of scope")


def validate_neo_persistent_complex_state(
    state: NeoPersistentComplexState,
    *,
    name: str = "state",
) -> None:
    """Validate the direct Phase-A/Phase-C split interface."""
    if (
        state.m0.ndim != 3
        or state.m0.shape[0] != FOCUS_COUNT
        or state.m0.shape[2] != M0_WIDTH
    ):
        raise ValueError(
            f"{name}.m0 must have shape (2,E,128), got {tuple(state.m0.shape)}"
        )
    _require_shape(
        f"{name}.m1",
        state.m1,
        (FOCUS_COUNT, state.edge_count, M1_WIDTH),
    )
    if state.edge_count <= 0:
        raise ValueError(f"{name} requires E > 0")
    if state.m0.dtype != torch.float32 or state.m1.dtype != torch.complex64:
        raise TypeError(f"{name} requires float32 m0 and complex64 m1")
    if state.m0.device != state.m1.device:
        raise ValueError(f"{name}.m0 and {name}.m1 must share a device")
    if not state.m0.is_contiguous() or not state.m1.is_contiguous():
        raise ValueError(f"{name} tensors must be focus-major contiguous")


def prepare_neo_persistent_complex_weights(
    w0: Tensor,
    wp: Tensor,
    gate: Tensor,
) -> NeoPersistentComplexWeights:
    """Convert exact block-real frozen weights to persistent complex weights.

    ``w0`` and ``wp`` use the live ``(input, output)`` orientation consumed by
    ``torch.bmm``.  The pair block must be exactly ``[[U,V],[-V,U]]``.
    """
    _require_shape("w0", w0, (STACK_LAYERS, FOCUS_COUNT, M0_WIDTH, M0_WIDTH))
    _require_shape(
        "wp",
        wp,
        (STACK_LAYERS, FOCUS_COUNT, PAIR_WIDTH, PAIR_WIDTH),
    )
    _require_shape(
        "gate",
        gate,
        (GATED_LAYERS, FOCUS_COUNT, CHANNELS, GATE_WIDTH),
    )
    if not w0.is_cuda:
        raise ValueError("weights must be CUDA tensors")
    device = w0.device
    for name, tensor in (("w0", w0), ("wp", wp), ("gate", gate)):
        _require_frozen_cuda_tensor(
            name,
            tensor,
            dtype=torch.float32,
            device=device,
        )

    u = wp[:, :, :M1_WIDTH, :M1_WIDTH]
    v = wp[:, :, :M1_WIDTH, M1_WIDTH:]
    if not torch.equal(wp[:, :, M1_WIDTH:, :M1_WIDTH], -v):
        raise ValueError("wp lower-left block must equal -V exactly")
    if not torch.equal(wp[:, :, M1_WIDTH:, M1_WIDTH:], u):
        raise ValueError("wp lower-right block must equal U exactly")

    w0_live = w0.detach().contiguous()
    wc_live = torch.complex(u, v).contiguous()
    w0_h = w0_live.transpose(-2, -1).contiguous()
    wc_h = wc_live.conj().transpose(-2, -1).contiguous()

    return NeoPersistentComplexWeights(
        w0=w0_live,
        wc=wc_live,
        w0_h=w0_h,
        wc_h=wc_h,
        gate=gate.detach().contiguous(),
    )


@cute.jit
def _sigmoid(value):
    return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-value))


@cute.jit
def _stage_gate_weight(gate_weight, shared_weight, focus, tidx):
    for load_slot in cutlass.range_constexpr(WEIGHT_LOADS_PER_THREAD):
        linear = tidx + load_slot * THREADS
        source_channel = linear // GATE_WIDTH
        gate_channel = linear - source_channel * GATE_WIDTH
        shared_weight[source_channel * WEIGHT_SMEM_STRIDE + gate_channel] = gate_weight[
            focus, source_channel, gate_channel
        ].to(cutlass.Float32)


@cute.jit
def _load_gate_values(shared_weight, scalar_rows, row_slot, channel):
    gate0_logit = cutlass.Float32(0.0)
    gate1_logit = cutlass.Float32(0.0)
    gate2_logit = cutlass.Float32(0.0)
    scalar_base = row_slot * CHANNELS
    for source_channel in cutlass.range_constexpr(CHANNELS):
        source = scalar_rows[scalar_base + source_channel]
        weight_base = source_channel * WEIGHT_SMEM_STRIDE + channel
        gate0_logit += source * shared_weight[weight_base]
        gate1_logit += source * shared_weight[weight_base + CHANNELS]
        gate2_logit += source * shared_weight[weight_base + 2 * CHANNELS]
    return _sigmoid(gate0_logit), _sigmoid(gate1_logit), _sigmoid(gate2_logit)


@cute.jit
def _select_gate(gate0, gate1, gate2, group):
    gate = gate0
    if cutlass.const_expr(group == 1):
        gate = gate1
    if cutlass.const_expr(group == 2):
        gate = gate2
    return gate


@cute.jit
def _persistent_gate_forward_jit(
    residual0: cute.Tensor,
    residual1_ri: cute.Tensor,
    z0: cute.Tensor,
    z1_ri: cute.Tensor,
    gate_weight: cute.Tensor,
    out0: cute.Tensor,
    out1_ri: cute.Tensor,
    stream: CUstream,
):
    edge_count = z0.shape[1]
    _persistent_gate_forward_kernel(
        residual0,
        residual1_ri,
        z0,
        z1_ri,
        gate_weight,
        out0,
        out1_ri,
    ).launch(
        grid=[cute.ceil_div(edge_count, ROWS_PER_BLOCK), FOCUS_COUNT, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _persistent_gate_forward_kernel(
    residual0: cute.Tensor,
    residual1_ri: cute.Tensor,
    z0: cute.Tensor,
    z1_ri: cute.Tensor,
    gate_weight: cute.Tensor,
    out0: cute.Tensor,
    out1_ri: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    edge_block, focus, _ = cute.arch.block_idx()
    row_slot = tidx // CHANNELS
    channel = tidx - row_slot * CHANNELS
    edge = edge_block * ROWS_PER_BLOCK + row_slot
    edge_count = z0.shape[1]

    smem = cutlass.utils.SmemAllocator()
    shared_weight = smem.allocate_tensor(
        cutlass.Float32,
        CHANNELS * WEIGHT_SMEM_STRIDE,
    )
    scalar_rows = smem.allocate_tensor(
        cutlass.Float32,
        ROWS_PER_BLOCK * CHANNELS,
    )
    _stage_gate_weight(gate_weight, shared_weight, focus, tidx)
    scalar = cutlass.Float32(0.0)
    if edge < edge_count:
        scalar = z0[focus, edge, channel].to(cutlass.Float32)
    scalar_rows[row_slot * CHANNELS + channel] = scalar
    cute.arch.sync_threads()

    if edge < edge_count:
        gate0, gate1, gate2 = _load_gate_values(
            shared_weight,
            scalar_rows,
            row_slot,
            channel,
        )
        out0[focus, edge, channel] = (
            residual0[focus, edge, channel].to(cutlass.Float32)
            + scalar * _sigmoid(scalar)
        ).to(out0.element_type)
        for group in cutlass.range_constexpr(GATE_GROUPS):
            gate = _select_gate(gate0, gate1, gate2, group)
            m0_col = (group + 1) * CHANNELS + channel
            out0[focus, edge, m0_col] = (
                residual0[focus, edge, m0_col].to(cutlass.Float32)
                + z0[focus, edge, m0_col].to(cutlass.Float32) * gate
            ).to(out0.element_type)
            m1_col = group * CHANNELS + channel
            for component in cutlass.range_constexpr(2):
                out1_ri[focus, edge, m1_col, component] = (
                    residual1_ri[focus, edge, m1_col, component].to(cutlass.Float32)
                    + z1_ri[focus, edge, m1_col, component].to(cutlass.Float32) * gate
                ).to(out1_ri.element_type)


@cute.jit
def _persistent_gate_adjoint_jit(
    grad0: cute.Tensor,
    grad1_ri: cute.Tensor,
    z0: cute.Tensor,
    z1_ri: cute.Tensor,
    gate_weight: cute.Tensor,
    grad_z0: cute.Tensor,
    grad_z1_ri: cute.Tensor,
    stream: CUstream,
):
    edge_count = z0.shape[1]
    _persistent_gate_adjoint_kernel(
        grad0,
        grad1_ri,
        z0,
        z1_ri,
        gate_weight,
        grad_z0,
        grad_z1_ri,
    ).launch(
        grid=[cute.ceil_div(edge_count, ROWS_PER_BLOCK), FOCUS_COUNT, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _persistent_gate_adjoint_kernel(
    grad0: cute.Tensor,
    grad1_ri: cute.Tensor,
    z0: cute.Tensor,
    z1_ri: cute.Tensor,
    gate_weight: cute.Tensor,
    grad_z0: cute.Tensor,
    grad_z1_ri: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    edge_block, focus, _ = cute.arch.block_idx()
    row_slot = tidx // CHANNELS
    channel = tidx - row_slot * CHANNELS
    edge = edge_block * ROWS_PER_BLOCK + row_slot
    edge_count = z0.shape[1]

    smem = cutlass.utils.SmemAllocator()
    shared_weight = smem.allocate_tensor(
        cutlass.Float32,
        CHANNELS * WEIGHT_SMEM_STRIDE,
    )
    scalar_rows = smem.allocate_tensor(
        cutlass.Float32,
        ROWS_PER_BLOCK * CHANNELS,
    )
    grad_logits = smem.allocate_tensor(
        cutlass.Float32,
        ROWS_PER_BLOCK * GATE_WIDTH,
    )
    _stage_gate_weight(gate_weight, shared_weight, focus, tidx)
    scalar = cutlass.Float32(0.0)
    if edge < edge_count:
        scalar = z0[focus, edge, channel].to(cutlass.Float32)
    scalar_rows[row_slot * CHANNELS + channel] = scalar
    cute.arch.sync_threads()

    grad_scalar = cutlass.Float32(0.0)
    grad_logit0 = cutlass.Float32(0.0)
    grad_logit1 = cutlass.Float32(0.0)
    grad_logit2 = cutlass.Float32(0.0)
    if edge < edge_count:
        gate0, gate1, gate2 = _load_gate_values(
            shared_weight,
            scalar_rows,
            row_slot,
            channel,
        )
        scalar_sigmoid = _sigmoid(scalar)
        grad_scalar = (
            grad0[focus, edge, channel].to(cutlass.Float32)
            * scalar_sigmoid
            * (cutlass.Float32(1.0) + scalar * (cutlass.Float32(1.0) - scalar_sigmoid))
        )

        for group in cutlass.range_constexpr(GATE_GROUPS):
            gate = _select_gate(gate0, gate1, gate2, group)
            m0_col = (group + 1) * CHANNELS + channel
            upstream0 = grad0[focus, edge, m0_col].to(cutlass.Float32)
            value0 = z0[focus, edge, m0_col].to(cutlass.Float32)
            grad_z0[focus, edge, m0_col] = (upstream0 * gate).to(grad_z0.element_type)
            contribution = upstream0 * value0

            m1_col = group * CHANNELS + channel
            for component in cutlass.range_constexpr(2):
                upstream1 = grad1_ri[focus, edge, m1_col, component].to(cutlass.Float32)
                value1 = z1_ri[focus, edge, m1_col, component].to(cutlass.Float32)
                grad_z1_ri[focus, edge, m1_col, component] = (upstream1 * gate).to(
                    grad_z1_ri.element_type
                )
                contribution += upstream1 * value1

            gate_derivative = gate * (cutlass.Float32(1.0) - gate)
            if cutlass.const_expr(group == 0):
                grad_logit0 = contribution * gate_derivative
            if cutlass.const_expr(group == 1):
                grad_logit1 = contribution * gate_derivative
            if cutlass.const_expr(group == 2):
                grad_logit2 = contribution * gate_derivative

    grad_base = row_slot * GATE_WIDTH + channel
    grad_logits[grad_base] = grad_logit0
    grad_logits[grad_base + CHANNELS] = grad_logit1
    grad_logits[grad_base + 2 * CHANNELS] = grad_logit2
    cute.arch.sync_threads()

    if edge < edge_count:
        for gate_channel in cutlass.range_constexpr(GATE_WIDTH):
            grad_scalar += (
                grad_logits[row_slot * GATE_WIDTH + gate_channel]
                * shared_weight[channel * WEIGHT_SMEM_STRIDE + gate_channel]
            )
        grad_z0[focus, edge, channel] = grad_scalar.to(grad_z0.element_type)


def _fake_m0():
    edge_count = cute.sym_int64()
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, edge_count, M0_WIDTH),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_m1_ri():
    edge_count = cute.sym_int64()
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, edge_count, M1_WIDTH, 2),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_gate_weight():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, CHANNELS, GATE_WIDTH),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _compile_forward() -> Callable:
    return cute.compile(
        _persistent_gate_forward_jit,
        _fake_m0(),
        _fake_m1_ri(),
        _fake_m0(),
        _fake_m1_ri(),
        _fake_gate_weight(),
        _fake_m0(),
        _fake_m1_ri(),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _compile_adjoint() -> Callable:
    return cute.compile(
        _persistent_gate_adjoint_jit,
        _fake_m0(),
        _fake_m1_ri(),
        _fake_m0(),
        _fake_m1_ri(),
        _fake_gate_weight(),
        _fake_m0(),
        _fake_m1_ri(),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@device_aware_lru_cache(maxsize=2)
def _compiled_forward() -> Callable:
    return _compile_forward()


@device_aware_lru_cache(maxsize=2)
def _compiled_adjoint() -> Callable:
    return _compile_adjoint()


def _m1_real_view(m1: Tensor) -> Tensor:
    view = torch.view_as_real(m1)
    if not view.is_contiguous():
        raise ValueError("complex state must expose a contiguous real/imag view")
    return view


def _run_gate_forward(
    residual: NeoPersistentComplexState,
    z: NeoPersistentComplexState,
    gate_weight: Tensor,
    out: NeoPersistentComplexState,
) -> None:
    with torch.cuda.device(z.m0.device):
        _compiled_forward()(
            residual.m0,
            _m1_real_view(residual.m1),
            z.m0,
            _m1_real_view(z.m1),
            gate_weight,
            out.m0,
            _m1_real_view(out.m1),
        )


def _run_gate_adjoint(
    grad: NeoPersistentComplexState,
    z: NeoPersistentComplexState,
    gate_weight: Tensor,
    out: NeoPersistentComplexState,
) -> None:
    with torch.cuda.device(z.m0.device):
        _compiled_adjoint()(
            grad.m0,
            _m1_real_view(grad.m1),
            z.m0,
            _m1_real_view(z.m1),
            gate_weight,
            out.m0,
            _m1_real_view(out.m1),
        )


def _empty_state_like(state: NeoPersistentComplexState) -> NeoPersistentComplexState:
    return NeoPersistentComplexState(
        m0=torch.empty_like(state.m0),
        m1=torch.empty_like(state.m1),
    )
