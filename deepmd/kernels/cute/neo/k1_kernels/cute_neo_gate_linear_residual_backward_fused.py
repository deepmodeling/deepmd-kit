# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CuTe Neo gate-linear + gate/residual backward without saved logits."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from ..compile_cache import (
    device_aware_lru_cache,
)

# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN201, ANN202, TC002


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}
ROWS_PER_BLOCK = 2


@cute.jit
def _sigmoid(value):
    return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-value))


@cute.jit
def neo_gate_linear_residual_backward_fused_jit(
    grad_out: cute.Tensor,
    y: cute.Tensor,
    gate_weight: cute.Tensor,
    grad_y: cute.Tensor,
    stream: CUstream,
    rows_per_block: cutlass.Constexpr[int],
):
    rows, _ = y.shape
    neo_gate_linear_residual_backward_fused_kernel(
        grad_out,
        y,
        gate_weight,
        grad_y,
        rows_per_block,
    ).launch(
        grid=[cute.ceil_div(rows, rows_per_block), 1, 1],
        block=[32 * rows_per_block, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_gate_linear_residual_backward_fused_kernel(
    grad_out: cute.Tensor,
    y: cute.Tensor,
    gate_weight: cute.Tensor,
    grad_y: cute.Tensor,
    rows_per_block: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    block_row, _, _ = cute.arch.block_idx()
    row_slot = tidx // 32
    channel = tidx - row_slot * 32
    row = block_row * rows_per_block + row_slot
    rows, _ = y.shape

    smem = cutlass.utils.SmemAllocator()
    grad_logits = smem.allocate_tensor(cutlass.Float32, rows_per_block * 3 * 32)
    smem_base = row_slot * 3 * 32

    for gate_degree in cutlass.range_constexpr(3):
        grad_logits[smem_base + gate_degree * 32 + channel] = cutlass.Float32(0.0)
    cute.arch.sync_threads()

    grad_l0 = cutlass.Float32(0.0)
    if row < rows:
        focus = row - (row // 2) * 2
        y0 = y[row, channel].to(cutlass.Float32)
        sig0 = _sigmoid(y0)
        grad0 = grad_out[row, channel].to(cutlass.Float32)
        grad_l0 = (
            grad0 * sig0 * (cutlass.Float32(1.0) + y0 * (cutlass.Float32(1.0) - sig0))
        )

        gate0_logit = cutlass.Float32(0.0)
        gate1_logit = cutlass.Float32(0.0)
        gate2_logit = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(32):
            src = y[row, k].to(cutlass.Float32)
            gate0_logit += src * gate_weight[k, focus, channel].to(cutlass.Float32)
            gate1_logit += src * gate_weight[k, focus, 32 + channel].to(cutlass.Float32)
            gate2_logit += src * gate_weight[k, focus, 64 + channel].to(cutlass.Float32)

        gate0 = _sigmoid(gate0_logit)
        gate1 = _sigmoid(gate1_logit)
        gate2 = _sigmoid(gate2_logit)
        for d in cutlass.range_constexpr(1, 10, 1):
            gate_idx = channel
            gate = gate0
            if cutlass.const_expr((d - 1) % 3 == 1):
                gate_idx = 32 + channel
                gate = gate1
            if cutlass.const_expr((d - 1) % 3 == 2):
                gate_idx = 64 + channel
                gate = gate2
            idx = d * 32 + channel
            gout = grad_out[row, idx].to(cutlass.Float32)
            yv = y[row, idx].to(cutlass.Float32)
            grad_y[row, idx] = (gout * gate).to(grad_y.element_type)
            old = grad_logits[smem_base + gate_idx]
            grad_logits[smem_base + gate_idx] = old + gout * yv * gate * (
                cutlass.Float32(1.0) - gate
            )
    cute.arch.sync_threads()

    if row < rows:
        focus = row - (row // 2) * 2
        grad_gate_src = cutlass.Float32(0.0)
        for out_idx in cutlass.range_constexpr(3 * 32):
            grad_gate_src += grad_logits[smem_base + out_idx] * gate_weight[
                channel, focus, out_idx
            ].to(cutlass.Float32)
        grad_y[row, channel] = (grad_l0 + grad_gate_src).to(grad_y.element_type)


@device_aware_lru_cache(maxsize=2)
def compile_neo_gate_linear_residual_backward_fused() -> Callable:
    rows = cute.sym_int64()
    fake_grad_out = make_fake_compact_tensor(
        cutlass.Float32, (rows, 10 * 32), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_y = make_fake_compact_tensor(
        cutlass.Float32, (rows, 10 * 32), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_gate_weight = make_fake_compact_tensor(
        cutlass.Float32, (32, 2, 3 * 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    fake_grad_y = make_fake_compact_tensor(
        cutlass.Float32, (rows, 10 * 32), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        neo_gate_linear_residual_backward_fused_jit,
        fake_grad_out,
        fake_y,
        fake_gate_weight,
        fake_grad_y,
        fake_stream,
        ROWS_PER_BLOCK,
        options="--enable-tvm-ffi",
    )
