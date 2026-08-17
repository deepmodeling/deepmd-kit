# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN201, ANN202, TC002, TC003
"""Strict-FP32 vectorized structural-gate forward/backward for Neo K1.

The split path keeps both neighboring SO2 products and the two gate projections
in cuBLAS. Its elementwise consumer assigns eight threads to each 32-channel
row, with each thread moving an aligned float4. Both directions preserve the
split-gate tensor contract and leave dense projections in PyTorch/cuBLAS.
"""

from __future__ import (
    annotations,
)

from collections.abc import (
    Callable,
)
from functools import (
    lru_cache,
)

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from .. import (
    runtime_policy,
)

FOCUS_COUNT = 2
REDUCED_COUNT = 10
CHANNELS = 32
GATE_COUNT = 3
VECTOR_WIDTH = 4
CHANNEL_GROUPS = CHANNELS // VECTOR_WIDTH
ROWS_PER_BLOCK = 16
THREADS = ROWS_PER_BLOCK * CHANNEL_GROUPS
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}
FORWARD_TENSOR_NAMES = ("residual", "y", "logits", "out")
BACKWARD_TENSOR_NAMES = ("grad_out", "y", "logits", "grad_y", "grad_logits")


def _guard_vec4_dispatch(
    kernel: Callable,
    tensor_names: tuple[str, ...],
) -> Callable:
    from ..k1_gate_structural import (
        _dispatch_aligned_vec4_kernel,
    )

    def dispatch(*tensors: object):
        return _dispatch_aligned_vec4_kernel(kernel, tensor_names, *tensors)

    return dispatch


@cute.jit
def _sigmoid(value):
    return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-value))


@cute.jit
def neo_gate_split_structural_vec4_sm80_forward_jit(
    residual: cute.Tensor,
    y: cute.Tensor,
    logits: cute.Tensor,
    out: cute.Tensor,
    stream: CUstream,
):
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        residual.element_type,
        num_bits_per_copy=residual.element_type.width * VECTOR_WIDTH,
    )
    channel_thread_layout = cute.make_ordered_layout(
        (1, CHANNEL_GROUPS),
        order=(1, 0),
    )
    channel_value_layout = cute.make_ordered_layout(
        (1, VECTOR_WIDTH),
        order=(1, 0),
    )
    vector_copy = cute.make_tiled_copy_tv(
        copy_atom,
        channel_thread_layout,
        channel_value_layout,
    )
    channel_layout = cute.make_layout((1, CHANNELS), stride=(CHANNELS, 1))
    rows, _ = y.shape
    neo_gate_split_structural_vec4_sm80_forward_kernel(
        residual,
        y,
        logits,
        out,
        channel_layout,
        vector_copy,
    ).launch(
        grid=[cute.ceil_div(rows, ROWS_PER_BLOCK), 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_gate_split_structural_vec4_sm80_forward_kernel(
    residual: cute.Tensor,
    y: cute.Tensor,
    logits: cute.Tensor,
    out: cute.Tensor,
    channel_layout: cute.Layout,
    vector_copy: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    block_row, _, _ = cute.arch.block_idx()
    row_slot = tidx // CHANNEL_GROUPS
    channel_group = tidx - row_slot * CHANNEL_GROUPS
    row = block_row * ROWS_PER_BLOCK + row_slot
    rows, _ = y.shape

    if row < rows:
        edge = row // FOCUS_COUNT
        focus = row - edge * FOCUS_COUNT
        thread_copy = vector_copy.get_slice(channel_group)

        y0_tile = cute.local_tile(
            y,
            tiler=(1, CHANNELS),
            coord=(row, 0),
        )
        residual0_tile = cute.local_tile(
            residual,
            tiler=(1, CHANNELS),
            coord=(row, 0),
        )
        out0_tile = cute.local_tile(
            out,
            tiler=(1, CHANNELS),
            coord=(row, 0),
        )
        thread_y = thread_copy.partition_S(y0_tile)
        thread_residual = thread_copy.partition_S(residual0_tile)
        thread_out = thread_copy.partition_D(out0_tile)
        y_fragment = cute.make_fragment_like(thread_y, cutlass.Float32)
        residual_fragment = cute.make_fragment_like(
            thread_residual,
            cutlass.Float32,
        )
        cute.copy(vector_copy, thread_y, y_fragment)
        cute.copy(vector_copy, thread_residual, residual_fragment)
        for value_idx in cutlass.range_constexpr(VECTOR_WIDTH):
            y0 = y_fragment[value_idx].to(cutlass.Float32)
            value = y0 * _sigmoid(y0)
            value += residual_fragment[value_idx].to(cutlass.Float32)
            residual_fragment[value_idx] = value
        cute.copy(vector_copy, residual_fragment, thread_out)

        for gate_index in cutlass.range_constexpr(GATE_COUNT):
            logits_tile = cute.local_tile(
                logits,
                tiler=(1, 1, CHANNELS),
                coord=(focus, edge, gate_index),
            )
            logits_panel = cute.make_tensor(logits_tile.iterator, channel_layout)
            thread_logits = thread_copy.partition_S(logits_panel)
            gate_fragment = cute.make_fragment_like(
                thread_logits,
                cutlass.Float32,
            )
            cute.copy(vector_copy, thread_logits, gate_fragment)
            for value_idx in cutlass.range_constexpr(VECTOR_WIDTH):
                gate_fragment[value_idx] = _sigmoid(
                    gate_fragment[value_idx].to(cutlass.Float32)
                )

            for repeat in cutlass.range_constexpr(3):
                degree = 1 + gate_index + repeat * GATE_COUNT
                y_tile = cute.local_tile(
                    y,
                    tiler=(1, CHANNELS),
                    coord=(row, degree),
                )
                residual_tile = cute.local_tile(
                    residual,
                    tiler=(1, CHANNELS),
                    coord=(row, degree),
                )
                out_tile = cute.local_tile(
                    out,
                    tiler=(1, CHANNELS),
                    coord=(row, degree),
                )
                thread_y = thread_copy.partition_S(y_tile)
                thread_residual = thread_copy.partition_S(residual_tile)
                thread_out = thread_copy.partition_D(out_tile)
                y_fragment = cute.make_fragment_like(thread_y, cutlass.Float32)
                residual_fragment = cute.make_fragment_like(
                    thread_residual,
                    cutlass.Float32,
                )
                cute.copy(vector_copy, thread_y, y_fragment)
                cute.copy(vector_copy, thread_residual, residual_fragment)
                for value_idx in cutlass.range_constexpr(VECTOR_WIDTH):
                    value = y_fragment[value_idx].to(cutlass.Float32) * gate_fragment[
                        value_idx
                    ].to(cutlass.Float32)
                    value += residual_fragment[value_idx].to(cutlass.Float32)
                    residual_fragment[value_idx] = value
                cute.copy(vector_copy, residual_fragment, thread_out)


@cute.jit
def neo_gate_split_structural_vec4_sm80_backward_jit(
    grad_out: cute.Tensor,
    y: cute.Tensor,
    logits: cute.Tensor,
    grad_y: cute.Tensor,
    grad_logits: cute.Tensor,
    stream: CUstream,
):
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        y.element_type,
        num_bits_per_copy=y.element_type.width * VECTOR_WIDTH,
    )
    channel_thread_layout = cute.make_ordered_layout(
        (1, CHANNEL_GROUPS),
        order=(1, 0),
    )
    channel_value_layout = cute.make_ordered_layout(
        (1, VECTOR_WIDTH),
        order=(1, 0),
    )
    vector_copy = cute.make_tiled_copy_tv(
        copy_atom,
        channel_thread_layout,
        channel_value_layout,
    )
    channel_layout = cute.make_layout((1, CHANNELS), stride=(CHANNELS, 1))
    rows, _ = y.shape
    neo_gate_split_structural_vec4_sm80_backward_kernel(
        grad_out,
        y,
        logits,
        grad_y,
        grad_logits,
        channel_layout,
        vector_copy,
    ).launch(
        grid=[cute.ceil_div(rows, ROWS_PER_BLOCK), 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_gate_split_structural_vec4_sm80_backward_kernel(
    grad_out: cute.Tensor,
    y: cute.Tensor,
    logits: cute.Tensor,
    grad_y: cute.Tensor,
    grad_logits: cute.Tensor,
    channel_layout: cute.Layout,
    vector_copy: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    block_row, _, _ = cute.arch.block_idx()
    row_slot = tidx // CHANNEL_GROUPS
    channel_group = tidx - row_slot * CHANNEL_GROUPS
    row = block_row * ROWS_PER_BLOCK + row_slot
    rows, _ = y.shape

    if row < rows:
        edge = row // FOCUS_COUNT
        focus = row - edge * FOCUS_COUNT
        thread_copy = vector_copy.get_slice(channel_group)

        y0_tile = cute.local_tile(y, tiler=(1, CHANNELS), coord=(row, 0))
        grad_y0_tile = cute.local_tile(
            grad_y,
            tiler=(1, CHANNELS),
            coord=(row, 0),
        )
        grad_out0_panel = cute.local_tile(
            grad_out,
            tiler=(1, CHANNELS),
            coord=(row, 0),
        )
        thread_y0 = thread_copy.partition_S(y0_tile)
        thread_grad_out0 = thread_copy.partition_S(grad_out0_panel)
        thread_grad_y0 = thread_copy.partition_D(grad_y0_tile)
        y0_fragment = cute.make_fragment_like(thread_y0, cutlass.Float32)
        grad_out0_fragment = cute.make_fragment_like(
            thread_grad_out0,
            cutlass.Float32,
        )
        cute.copy(vector_copy, thread_y0, y0_fragment)
        cute.copy(vector_copy, thread_grad_out0, grad_out0_fragment)
        for value_idx in cutlass.range_constexpr(VECTOR_WIDTH):
            y0 = y0_fragment[value_idx].to(cutlass.Float32)
            sig0 = _sigmoid(y0)
            grad0 = grad_out0_fragment[value_idx].to(cutlass.Float32)
            y0_fragment[value_idx] = (
                grad0
                * sig0
                * (cutlass.Float32(1.0) + y0 * (cutlass.Float32(1.0) - sig0))
            )
        cute.copy(vector_copy, y0_fragment, thread_grad_y0)

        for gate_index in cutlass.range_constexpr(GATE_COUNT):
            logits_tile = cute.local_tile(
                logits,
                tiler=(1, 1, CHANNELS),
                coord=(focus, edge, gate_index),
            )
            logits_panel = cute.make_tensor(logits_tile.iterator, channel_layout)
            thread_logits = thread_copy.partition_S(logits_panel)
            gate_fragment = cute.make_fragment_like(
                thread_logits,
                cutlass.Float32,
            )
            grad_logit_fragment = cute.make_fragment_like(
                thread_logits,
                cutlass.Float32,
            )
            cute.copy(vector_copy, thread_logits, gate_fragment)
            for value_idx in cutlass.range_constexpr(VECTOR_WIDTH):
                gate_fragment[value_idx] = _sigmoid(
                    gate_fragment[value_idx].to(cutlass.Float32)
                )
                grad_logit_fragment[value_idx] = cutlass.Float32(0.0)

            for repeat in cutlass.range_constexpr(3):
                degree = 1 + gate_index + repeat * GATE_COUNT
                y_tile = cute.local_tile(
                    y,
                    tiler=(1, CHANNELS),
                    coord=(row, degree),
                )
                grad_y_tile = cute.local_tile(
                    grad_y,
                    tiler=(1, CHANNELS),
                    coord=(row, degree),
                )
                grad_out_panel = cute.local_tile(
                    grad_out,
                    tiler=(1, CHANNELS),
                    coord=(row, degree),
                )
                thread_y = thread_copy.partition_S(y_tile)
                thread_grad_out = thread_copy.partition_S(grad_out_panel)
                thread_grad_y = thread_copy.partition_D(grad_y_tile)
                y_fragment = cute.make_fragment_like(thread_y, cutlass.Float32)
                grad_out_fragment = cute.make_fragment_like(
                    thread_grad_out,
                    cutlass.Float32,
                )
                cute.copy(vector_copy, thread_y, y_fragment)
                cute.copy(vector_copy, thread_grad_out, grad_out_fragment)
                for value_idx in cutlass.range_constexpr(VECTOR_WIDTH):
                    gate = gate_fragment[value_idx].to(cutlass.Float32)
                    gout = grad_out_fragment[value_idx].to(cutlass.Float32)
                    y_value = y_fragment[value_idx].to(cutlass.Float32)
                    y_fragment[value_idx] = gout * gate
                    grad_logit_fragment[value_idx] += (
                        gout * y_value * gate * (cutlass.Float32(1.0) - gate)
                    )
                cute.copy(vector_copy, y_fragment, thread_grad_y)

            grad_logits_tile = cute.local_tile(
                grad_logits,
                tiler=(1, 1, CHANNELS),
                coord=(focus, edge, gate_index),
            )
            grad_logits_panel = cute.make_tensor(
                grad_logits_tile.iterator,
                channel_layout,
            )
            thread_grad_logits = thread_copy.partition_D(grad_logits_panel)
            cute.copy(vector_copy, grad_logit_fragment, thread_grad_logits)


@lru_cache(maxsize=8)
def compile_neo_gate_split_structural_vec4_sm80_forward(
    compile_identity: tuple[int, int, int] | None = None,
) -> Callable:
    if (
        compile_identity is not None
        and compile_identity[1:] not in runtime_policy.SUPPORTED_K1_CAPABILITIES
    ):
        raise ValueError(
            "vectorized structural gate forward requires a supported K1 device"
        )
    rows = cute.sym_int64()
    edges = cute.sym_int64()
    fake_residual = make_fake_compact_tensor(
        cutlass.Float32,
        (rows, REDUCED_COUNT * CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_y = make_fake_compact_tensor(
        cutlass.Float32,
        (rows, REDUCED_COUNT * CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_logits = make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, edges, GATE_COUNT * CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_out = make_fake_compact_tensor(
        cutlass.Float32,
        (rows, REDUCED_COUNT * CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return _guard_vec4_dispatch(
        cute.compile(
            neo_gate_split_structural_vec4_sm80_forward_jit,
            fake_residual,
            fake_y,
            fake_logits,
            fake_out,
            fake_stream,
            options="--enable-tvm-ffi",
        ),
        FORWARD_TENSOR_NAMES,
    )


@lru_cache(maxsize=8)
def compile_neo_gate_split_structural_vec4_sm80_backward(
    compile_identity: tuple[int, int, int] | None = None,
) -> Callable:
    if (
        compile_identity is not None
        and compile_identity[1:] not in runtime_policy.SUPPORTED_K1_CAPABILITIES
    ):
        raise ValueError(
            "vectorized structural gate backward requires a supported K1 device"
        )
    rows = cute.sym_int64()
    edges = cute.sym_int64()
    fake_grad_out = make_fake_compact_tensor(
        cutlass.Float32,
        (rows, REDUCED_COUNT * CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_y = make_fake_compact_tensor(
        cutlass.Float32,
        (rows, REDUCED_COUNT * CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_logits = make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, edges, GATE_COUNT * CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_y = make_fake_compact_tensor(
        cutlass.Float32,
        (rows, REDUCED_COUNT * CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_logits = make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, edges, GATE_COUNT * CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return _guard_vec4_dispatch(
        cute.compile(
            neo_gate_split_structural_vec4_sm80_backward_jit,
            fake_grad_out,
            fake_y,
            fake_logits,
            fake_grad_y,
            fake_grad_logits,
            fake_stream,
            options="--enable-tvm-ffi",
        ),
        BACKWARD_TENSOR_NAMES,
    )
