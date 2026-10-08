# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN201, ANN202, TC002, UP035
"""Fused Neo Q/K edge logits and first-backward input adjoints."""

from __future__ import (
    annotations,
)

from functools import (
    lru_cache,
)
from typing import (
    Callable,
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

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@cute.kernel
def neo_qk_edge_forward_kernel(
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    radial_l0: cute.Tensor,
    attention_weight: cute.Tensor,
    src: cute.Tensor,
    dst: cute.Tensor,
    logits: cute.Tensor,
    scale: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    block, focus, _ = cute.arch.block_idx()
    edge = block * 128 + tidx
    edges, _ = logits.shape
    if edge < edges:
        src_node = src[edge]
        dst_node = dst[edge]
        qk_acc = cutlass.Float32(0.0)
        radial_acc = cutlass.Float32(0.0)
        for channel in cutlass.range_constexpr(32):
            qk_acc += q_node[dst_node, focus, channel].to(cutlass.Float32) * k_node[
                src_node, focus, channel
            ].to(cutlass.Float32)
            radial_acc += radial_l0[edge, focus, channel].to(
                cutlass.Float32
            ) * attention_weight[channel, focus, 0].to(cutlass.Float32)
        logits[edge, focus] = (qk_acc * cutlass.Float32(scale) + radial_acc).to(
            logits.element_type
        )


@cute.jit
def neo_qk_edge_forward_jit(
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    radial_l0: cute.Tensor,
    attention_weight: cute.Tensor,
    src: cute.Tensor,
    dst: cute.Tensor,
    logits: cute.Tensor,
    stream: CUstream,
    scale: cutlass.Constexpr[float],
):
    edges, _ = logits.shape
    neo_qk_edge_forward_kernel(
        q_node,
        k_node,
        radial_l0,
        attention_weight,
        src,
        dst,
        logits,
        scale,
    ).launch(
        grid=[cute.ceil_div(edges, 128), 2, 1],
        block=[128, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_qk_edge_backward_kernel(
    grad_logits: cute.Tensor,
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    src: cute.Tensor,
    dst: cute.Tensor,
    grad_q_node: cute.Tensor,
    grad_k_node: cute.Tensor,
    scale: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    block, focus, _ = cute.arch.block_idx()
    edge = block * 8 + tidx // 32
    channel = tidx % 32
    edges, _ = grad_logits.shape
    if edge < edges:
        src_node = src[edge]
        dst_node = dst[edge]
        grad = grad_logits[edge, focus].to(cutlass.Float32) * cutlass.Float32(scale)
        grad_q = grad * k_node[src_node, focus, channel].to(cutlass.Float32)
        grad_k = grad * q_node[dst_node, focus, channel].to(cutlass.Float32)
        q_offset = (dst_node * 2 + focus) * 32 + channel
        k_offset = (src_node * 2 + focus) * 32 + channel
        q_ptr = grad_q_node.iterator + q_offset
        k_ptr = grad_k_node.iterator + k_offset
        cute.arch.atomic_add(q_ptr.llvm_ptr, grad_q, sem="relaxed", scope="gpu")
        cute.arch.atomic_add(k_ptr.llvm_ptr, grad_k, sem="relaxed", scope="gpu")


@cute.jit
def neo_qk_edge_backward_jit(
    grad_logits: cute.Tensor,
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    src: cute.Tensor,
    dst: cute.Tensor,
    grad_q_node: cute.Tensor,
    grad_k_node: cute.Tensor,
    stream: CUstream,
    scale: cutlass.Constexpr[float],
):
    edges, _ = grad_logits.shape
    neo_qk_edge_backward_kernel(
        grad_logits,
        q_node,
        k_node,
        src,
        dst,
        grad_q_node,
        grad_k_node,
        scale,
    ).launch(
        grid=[cute.ceil_div(edges, 8), 2, 1],
        block=[256, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_qk_node_input_adjoint_kernel(
    x_l0: cute.Tensor,
    grad_q_node: cute.Tensor,
    grad_k_node: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    norm_scale: cute.Tensor,
    grad_x_wide: cute.Tensor,
    eps: cutlass.Float32,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    node_in_block = tid // 64
    local_tid = tid % 64
    focus = local_tid // 32
    channel = local_tid % 32
    node = block * 4 + node_in_block
    nodes, _, _ = x_l0.shape

    if node < nodes:
        for flat_index in cutlass.range(local_tid, 16 * 64, 64):
            grad_x_wide[node, flat_index] = cutlass.Float32(0.0)

        grad_norm = cutlass.Float32(0.0)
        for output_channel in cutlass.range_constexpr(32):
            grad_norm += grad_q_node[node, focus, output_channel].to(
                cutlass.Float32
            ) * q_weight[channel, focus, output_channel].to(cutlass.Float32)
            grad_norm += grad_k_node[node, focus, output_channel].to(
                cutlass.Float32
            ) * k_weight[channel, focus, output_channel].to(cutlass.Float32)

        x = x_l0[node, focus, channel].to(cutlass.Float32)
        grad_scaled = grad_norm * norm_scale[focus, channel].to(cutlass.Float32)
        inv = cute.rsqrt(
            cute.arch.warp_reduction_sum(x * x) / cutlass.Float32(32.0) + eps
        )
        coeff = cute.arch.warp_reduction_sum(grad_scaled * x) / cutlass.Float32(32.0)
        grad_x = grad_scaled * inv - x * inv * inv * inv * coeff
        grad_x_wide[node, focus * 32 + channel] = grad_x.to(grad_x_wide.element_type)


@cute.jit
def neo_qk_node_input_adjoint_jit(
    x_l0: cute.Tensor,
    grad_q_node: cute.Tensor,
    grad_k_node: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    norm_scale: cute.Tensor,
    grad_x_wide: cute.Tensor,
    stream: CUstream,
    eps: cutlass.Float32,
):
    nodes, _, _ = x_l0.shape
    neo_qk_node_input_adjoint_kernel(
        x_l0,
        grad_q_node,
        grad_k_node,
        q_weight,
        k_weight,
        norm_scale,
        grad_x_wide,
        eps,
    ).launch(
        grid=[cute.ceil_div(nodes, 4), 1, 1],
        block=[256, 1, 1],
        stream=stream,
    )


def _fake_inputs():
    edge_count = cute.sym_int64()
    node_count = cute.sym_int64()
    q_node = make_fake_compact_tensor(
        cutlass.Float32, (node_count, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    k_node = make_fake_compact_tensor(
        cutlass.Float32, (node_count, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    radial = make_fake_compact_tensor(
        cutlass.Float32, (edge_count, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    weight = make_fake_compact_tensor(
        cutlass.Float32, (32, 2, 1), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    src = make_fake_compact_tensor(
        cutlass.Int32, (edge_count,), stride_order=(0,), **FAKE_TENSOR_KW
    )
    dst = make_fake_compact_tensor(
        cutlass.Int32, (edge_count,), stride_order=(0,), **FAKE_TENSOR_KW
    )
    logits = make_fake_compact_tensor(
        cutlass.Float32, (edge_count, 2), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    return q_node, k_node, radial, weight, src, dst, logits


@lru_cache(maxsize=8)
def compile_neo_qk_edge_forward(
    scale: float,
    compile_identity: tuple[int, int, int] | None = None,
) -> Callable:
    del compile_identity
    q_node, k_node, radial, weight, src, dst, logits = _fake_inputs()
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        neo_qk_edge_forward_jit,
        q_node,
        k_node,
        radial,
        weight,
        src,
        dst,
        logits,
        stream,
        scale,
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=8)
def compile_neo_qk_edge_backward(
    scale: float,
    compile_identity: tuple[int, int, int] | None = None,
) -> Callable:
    del compile_identity
    q_node, k_node, _radial, _weight, src, dst, logits = _fake_inputs()
    grad_q = make_fake_compact_tensor(
        cutlass.Float32, q_node.shape, stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    grad_k = make_fake_compact_tensor(
        cutlass.Float32, k_node.shape, stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        neo_qk_edge_backward_jit,
        logits,
        q_node,
        k_node,
        src,
        dst,
        grad_q,
        grad_k,
        stream,
        scale,
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=8)
def compile_neo_qk_node_input_adjoint(
    eps: float,
    compile_identity: tuple[int, int, int] | None = None,
) -> Callable:
    del compile_identity
    node_count = cute.sym_int64()
    x_l0 = make_fake_compact_tensor(
        cutlass.Float32, (node_count, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    grad_q = make_fake_compact_tensor(
        cutlass.Float32, x_l0.shape, stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    grad_k = make_fake_compact_tensor(
        cutlass.Float32, x_l0.shape, stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    q_weight = make_fake_compact_tensor(
        cutlass.Float32, (32, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    k_weight = make_fake_compact_tensor(
        cutlass.Float32, (32, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    norm_scale = make_fake_compact_tensor(
        cutlass.Float32, (2, 32), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    grad_x_wide = make_fake_compact_tensor(
        cutlass.Float32, (node_count, 16 * 64), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        neo_qk_node_input_adjoint_jit,
        x_l0,
        grad_q,
        grad_k,
        q_weight,
        k_weight,
        norm_scale,
        grad_x_wide,
        stream,
        cutlass.Float32(eps),
        options="--enable-tvm-ffi",
    )

    def run(
        x_l0_tensor,
        grad_q_tensor,
        grad_k_tensor,
        q_weight_tensor,
        k_weight_tensor,
        norm_scale_tensor,
        grad_x_wide_tensor,
    ):
        return compiled(
            x_l0_tensor,
            grad_q_tensor,
            grad_k_tensor,
            q_weight_tensor,
            k_weight_tensor,
            norm_scale_tensor,
            grad_x_wide_tensor,
            cutlass.Float32(eps),
        )

    return run
