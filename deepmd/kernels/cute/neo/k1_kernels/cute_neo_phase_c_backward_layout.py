# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exact-shape Neo Phase-C backward with fused layout-boundary reductions.

This kernel specializes ``D=16``, ``Dm=10``, ``F=2``, ``C=32`` and the
46-value packed Wigner panel. It keeps the destination adjoint in shared
memory once per node, but streams each degree once per edge into a lane-local
ten-value input-adjoint fragment. ``grad_Dt`` is reduced by the two focus
warps, so the effective stack input does not need a shared-memory slab.

The kernel also owns the two reductions immediately downstream of Phase C:

* attention ``grad_alpha`` is consumed in-place to produce envelope-softmax
  ``grad_logits``, ``grad_edge`` and ``grad_z``;
* focus ``grad_alpha`` is consumed in-place by the two-focus softmax/RMSNorm
  backward to produce ``grad_focus_src``.

The stack adjoint is written edge-major into the fully consumed final saved
activation, avoiding a separate edge-sized allocation. The focus-source
gradient remains focus-major.
"""

# ruff: noqa: ANN001, ANN201, ANN202, TC002

from __future__ import (
    annotations,
)

import operator
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
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

from ..k1_wigner_layout import PACKED_VALUE_COUNT as PACKED_WIGNER_VALUES

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


DEGREE_COUNT = 16
REDUCED_COUNT = 10
N_FOCUS = 2
FOCUS_CHANNELS = 32
HIDDEN = N_FOCUS * FOCUS_CHANNELS

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@dataclass(frozen=True)
class NeoPhaseCBackwardLayoutParams:
    grad_out: cute.Tensor
    stack: cute.Tensor
    wigner_dt: cute.Tensor
    alpha: cute.Tensor
    focus_alpha: cute.Tensor
    dst_ptr: cute.Tensor
    rotate_inv_rescale: cute.Tensor
    edge_gate: cute.Tensor
    z_bias_raw: cute.Tensor
    group_max: cute.Tensor
    denom: cute.Tensor
    focus_src: cute.Tensor
    focus_weight: cute.Tensor
    focus_scale: cute.Tensor
    grad_stack: cute.Tensor
    grad_wigner_dt: cute.Tensor
    grad_logits: cute.Tensor
    grad_edge: cute.Tensor
    grad_z_partial: cute.Tensor
    grad_z: cute.Tensor
    grad_focus_src: cute.Tensor


@cute.jit
def _warp_sum(value):
    return cute.arch.warp_reduction(value, operator.add)


@cute.jit
def _sigmoid(value):
    one = cutlass.Float32(1.0)
    return one / (one + cute.exp(-value))


@cute.jit
def _record_panel_term(
    panel_values,
    panel_offset,
    dt_partial,
    raw_fragment,
    grad_fragment,
    transformed,
    head,
    lane,
    reduced: cutlass.Constexpr[int],
    panel_index: cutlass.Constexpr[int],
    alpha_value,
    focus_value,
):
    """Consume one structural Wigner entry for grad-stack and grad-Dt."""
    panel = panel_values[panel_offset + panel_index]
    grad_fragment[reduced] = (
        grad_fragment[reduced].to(cutlass.Float32) + panel * transformed
    )

    effective_stack = raw_fragment[reduced].to(cutlass.Float32) * focus_value
    partial = _warp_sum(transformed * effective_stack * alpha_value)
    if lane == 0:
        dt_partial[head * PACKED_WIGNER_VALUES + panel_index] = partial


@cute.jit
def _focus_source_backward(
    params: NeoPhaseCBackwardLayoutParams,
    focus_grad,
    focus_inv,
    focus_grad_logits,
    focus_coeff,
    edge,
    head,
    lane,
    tidx,
    eps: cutlass.Constexpr[float],
    tau: cutlass.Constexpr[float],
    label_smoothing: cutlass.Constexpr[float],
):
    """Consume the Phase-C focus-alpha reduction without a global temporary."""
    value = params.focus_src[edge, head, lane].to(cutlass.Float32)
    scale = params.focus_scale[head, lane].to(cutlass.Float32)
    weight = params.focus_weight[lane, head].to(cutlass.Float32)

    square_sum = _warp_sum(value * value)
    if lane == 0:
        focus_inv[head] = cute.rsqrt(
            square_sum / cutlass.Float32(FOCUS_CHANNELS) + cutlass.Float32(eps)
        )

    if tidx == 0:
        probability_keep = cutlass.Float32(1.0 - label_smoothing)
        smooth = cutlass.Float32(label_smoothing / N_FOCUS)
        probability0 = (
            params.focus_alpha[edge, 0].to(cutlass.Float32) - smooth
        ) / probability_keep
        probability1 = (
            params.focus_alpha[edge, 1].to(cutlass.Float32) - smooth
        ) / probability_keep
        keep = cutlass.Float32(1.0 - label_smoothing)
        grad0 = focus_grad[0] * keep
        grad1 = focus_grad[1] * keep
        dot = grad0 * probability0 + grad1 * probability1
        inv_tau = cutlass.Float32(1.0 / tau)
        focus_grad_logits[0] = probability0 * (grad0 - dot) * inv_tau
        focus_grad_logits[1] = probability1 * (grad1 - dot) * inv_tau
    cute.arch.sync_threads()

    grad_scaled = focus_grad_logits[head] * weight * scale
    coeff_sum = _warp_sum(grad_scaled * value)
    if lane == 0:
        focus_coeff[head] = coeff_sum / cutlass.Float32(FOCUS_CHANNELS)
    cute.arch.sync_warp()

    inv = focus_inv[head]
    grad_value = grad_scaled * inv
    grad_value -= value * inv * inv * inv * focus_coeff[head]
    params.grad_focus_src[head, edge, lane] = grad_value.to(
        params.grad_focus_src.element_type
    )


@cute.kernel
def neo_phase_c_backward_layout_kernel(
    params: NeoPhaseCBackwardLayoutParams,
    raw_layout: cute.Layout,
    raw_tiled_copy: cute.TiledCopy,
    focus_eps: cutlass.Constexpr[float],
    focus_tau: cutlass.Constexpr[float],
    focus_label_smoothing: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    node, _, _ = cute.arch.block_idx()
    head = tidx // FOCUS_CHANNELS
    lane = tidx - head * FOCUS_CHANNELS
    lo = params.dst_ptr[node]
    hi = params.dst_ptr[node + 1]

    smem = cutlass.utils.SmemAllocator()
    t_values = smem.allocate_tensor(cutlass.Float32, DEGREE_COUNT * HIDDEN)
    panel_values = smem.allocate_tensor(
        cutlass.Float32,
        PACKED_WIGNER_VALUES,
    )
    dt_partial = smem.allocate_tensor(cutlass.Float32, N_FOCUS * PACKED_WIGNER_VALUES)
    focus_grad = smem.allocate_tensor(cutlass.Float32, N_FOCUS)
    focus_inv = smem.allocate_tensor(cutlass.Float32, N_FOCUS)
    focus_grad_logits = smem.allocate_tensor(cutlass.Float32, N_FOCUS)
    focus_coeff = smem.allocate_tensor(cutlass.Float32, N_FOCUS)
    softmax_dot_by_focus = smem.allocate_tensor(cutlass.Float32, N_FOCUS)
    gate_tile = smem.allocate_tensor(cutlass.Float32, HIDDEN)

    for degree in cutlass.range_constexpr(DEGREE_COUNT):
        index = degree * HIDDEN + tidx
        upstream = params.grad_out[node, degree, tidx].to(cutlass.Float32)
        rotate = params.rotate_inv_rescale[degree].to(cutlass.Float32)
        t_values[index] = upstream * rotate
    cute.arch.sync_threads()

    raw_thread_copy = raw_tiled_copy.get_slice(lane)
    softmax_dot = cutlass.Float32(0.0)

    for edge in cutlass.range(lo, hi, 1, unroll=1):
        stack_tile = cute.local_tile(
            params.stack,
            tiler=(1, 1, REDUCED_COUNT, FOCUS_CHANNELS),
            coord=(edge, head, 0, 0),
        )
        stack_head = cute.make_tensor(stack_tile.iterator, raw_layout)
        thread_stack = raw_thread_copy.partition_S(stack_head)
        raw_fragment = cute.make_fragment_like(thread_stack, cutlass.Float32)
        cute.copy(raw_tiled_copy, thread_stack, raw_fragment)
        if tidx < PACKED_WIGNER_VALUES:
            panel_values[tidx] = params.wigner_dt[edge, tidx].to(cutlass.Float32)
        cute.arch.sync_threads()

        alpha_value = params.alpha[edge, head].to(cutlass.Float32)
        focus_value = params.focus_alpha[edge, head].to(cutlass.Float32)
        grad_fragment = cute.make_fragment_like(raw_fragment, cutlass.Float32)
        grad_fragment.fill(0.0)

        transformed = t_values[tidx]
        _record_panel_term(
            panel_values,
            0,
            dt_partial,
            raw_fragment,
            grad_fragment,
            transformed,
            head,
            lane,
            0,
            0,
            alpha_value,
            focus_value,
        )
        for local_col in cutlass.range_constexpr(3):
            transformed = t_values[(1 + local_col) * HIDDEN + tidx]
            for row_slot in cutlass.range_constexpr(3):
                _record_panel_term(
                    panel_values,
                    0,
                    dt_partial,
                    raw_fragment,
                    grad_fragment,
                    transformed,
                    head,
                    lane,
                    1 + row_slot * 3,
                    1 + row_slot * 3 + local_col,
                    alpha_value,
                    focus_value,
                )
        for local_col in cutlass.range_constexpr(5):
            transformed = t_values[(4 + local_col) * HIDDEN + tidx]
            for row_slot in cutlass.range_constexpr(3):
                _record_panel_term(
                    panel_values,
                    0,
                    dt_partial,
                    raw_fragment,
                    grad_fragment,
                    transformed,
                    head,
                    lane,
                    2 + row_slot * 3,
                    10 + row_slot * 5 + local_col,
                    alpha_value,
                    focus_value,
                )
        for local_col in cutlass.range_constexpr(7):
            transformed = t_values[(9 + local_col) * HIDDEN + tidx]
            for row_slot in cutlass.range_constexpr(3):
                _record_panel_term(
                    panel_values,
                    0,
                    dt_partial,
                    raw_fragment,
                    grad_fragment,
                    transformed,
                    head,
                    lane,
                    3 + row_slot * 3,
                    25 + row_slot * 7 + local_col,
                    alpha_value,
                    focus_value,
                )

        grad_focus_part = cutlass.Float32(0.0)
        grad_alpha_part = cutlass.Float32(0.0)
        for reduced in cutlass.range_constexpr(REDUCED_COUNT):
            raw = raw_fragment[reduced].to(cutlass.Float32)
            grad_raw = grad_fragment[reduced].to(cutlass.Float32)
            grad_focus_part += grad_raw * raw * alpha_value
            grad_alpha_part += grad_raw * raw * focus_value
            grad_fragment[reduced] = grad_raw * focus_value * alpha_value

        # Every edge belongs to exactly one node CTA. Both focus warps have
        # loaded their complete source fragment, so this exact-address store
        # may reuse the fully consumed stack allocation.
        grad_stack_tile = cute.local_tile(
            params.grad_stack,
            tiler=(1, 1, REDUCED_COUNT, FOCUS_CHANNELS),
            coord=(edge, head, 0, 0),
        )
        grad_stack_head = cute.make_tensor(grad_stack_tile.iterator, raw_layout)
        thread_grad_stack = raw_thread_copy.partition_D(grad_stack_head)
        cute.copy(raw_tiled_copy, grad_fragment, thread_grad_stack)

        grad_focus_value = _warp_sum(grad_focus_part)
        grad_alpha_value = _warp_sum(grad_alpha_part)
        if lane == 0:
            focus_grad[head] = grad_focus_value
            # grad_logits is the dead grad-alpha slab during the first node pass.
            params.grad_logits[edge, head] = grad_alpha_value.to(
                params.grad_logits.element_type
            )
            softmax_dot += grad_alpha_value * alpha_value

        cute.arch.sync_threads()
        if tidx < PACKED_WIGNER_VALUES:
            value = dt_partial[tidx] + dt_partial[PACKED_WIGNER_VALUES + tidx]
            params.grad_wigner_dt[edge, tidx] = value.to(
                params.grad_wigner_dt.element_type
            )

        _focus_source_backward(
            params,
            focus_grad,
            focus_inv,
            focus_grad_logits,
            focus_coeff,
            edge,
            head,
            lane,
            tidx,
            focus_eps,
            focus_tau,
            focus_label_smoothing,
        )
        cute.arch.sync_threads()

    if lane == 0:
        softmax_dot_by_focus[head] = softmax_dot
        max_value = params.group_max[node, head].to(cutlass.Float32)
        denom = params.denom[node, head].to(cutlass.Float32)
        z_sigmoid = _sigmoid(params.z_bias_raw[head].to(cutlass.Float32))
        params.grad_z_partial[node, head] = (
            -softmax_dot * cute.exp(-max_value) / denom * z_sigmoid
        ).to(params.grad_z_partial.element_type)
    cute.arch.sync_threads()
    # The edge loop below is lane-striped, so every lane needs the node dot.
    softmax_dot = softmax_dot_by_focus[head]

    for edge_base in cutlass.range(lo, hi, FOCUS_CHANNELS, unroll=1):
        edge = edge_base + lane
        gate_contribution = cutlass.Float32(0.0)
        if edge < hi:
            upstream = params.grad_logits[edge, head].to(cutlass.Float32)
            centered = upstream - softmax_dot
            alpha_value = params.alpha[edge, head].to(cutlass.Float32)
            params.grad_logits[edge, head] = (alpha_value * centered).to(
                params.grad_logits.element_type
            )

            gate = params.edge_gate[edge].to(cutlass.Float32)
            if gate < cutlass.Float32(0.0):
                gate = cutlass.Float32(0.0)
            if gate > cutlass.Float32(0.0):
                gate_contribution = alpha_value * centered * cutlass.Float32(2.0) / gate
        gate_tile[tidx] = gate_contribution
        cute.arch.sync_threads()
        if head == 0 and edge < hi:
            params.grad_edge[edge] = (gate_tile[lane] + gate_tile[lane + 32]).to(
                params.grad_edge.element_type
            )
        cute.arch.sync_threads()


@cute.jit
def _cta_sum(value, scratch, threads: cutlass.Constexpr[int]):
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    warps = threads // 32
    value = _warp_sum(value)
    if lane == 0:
        scratch[warp] = value
    cute.arch.barrier()

    total = cutlass.Float32(0.0)
    if lane < warps:
        total = scratch[lane]
    return _warp_sum(total)


@cute.kernel
def neo_phase_c_backward_z_reduce_kernel(
    params: NeoPhaseCBackwardLayoutParams,
    threads: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    head, _, _ = cute.arch.block_idx()
    node_count, _ = params.grad_z_partial.shape
    warps = threads // 32

    smem = cutlass.utils.SmemAllocator()
    scratch = smem.allocate_tensor(cutlass.Float32, warps)
    local = cutlass.Float32(0.0)
    for node in cutlass.range(tidx, node_count, threads, unroll=1):
        local += params.grad_z_partial[node, head].to(cutlass.Float32)
    total = _cta_sum(local, scratch, threads)
    if tidx == 0:
        params.grad_z[head] = total.to(params.grad_z.element_type)


@cute.jit
def neo_phase_c_backward_layout_jit(
    grad_out: cute.Tensor,
    stack: cute.Tensor,
    wigner_dt: cute.Tensor,
    alpha: cute.Tensor,
    focus_alpha: cute.Tensor,
    dst_ptr: cute.Tensor,
    rotate_inv_rescale: cute.Tensor,
    edge_gate: cute.Tensor,
    z_bias_raw: cute.Tensor,
    group_max: cute.Tensor,
    denom: cute.Tensor,
    focus_src: cute.Tensor,
    focus_weight: cute.Tensor,
    focus_scale: cute.Tensor,
    grad_stack: cute.Tensor,
    grad_wigner_dt: cute.Tensor,
    grad_logits: cute.Tensor,
    grad_edge: cute.Tensor,
    grad_z_partial: cute.Tensor,
    grad_z: cute.Tensor,
    grad_focus_src: cute.Tensor,
    stream: CUstream,
    focus_eps: cutlass.Constexpr[float],
    focus_tau: cutlass.Constexpr[float],
    focus_label_smoothing: cutlass.Constexpr[float],
):
    params = NeoPhaseCBackwardLayoutParams(
        grad_out=grad_out,
        stack=stack,
        wigner_dt=wigner_dt,
        alpha=alpha,
        focus_alpha=focus_alpha,
        dst_ptr=dst_ptr,
        rotate_inv_rescale=rotate_inv_rescale,
        edge_gate=edge_gate,
        z_bias_raw=z_bias_raw,
        group_max=group_max,
        denom=denom,
        focus_src=focus_src,
        focus_weight=focus_weight,
        focus_scale=focus_scale,
        grad_stack=grad_stack,
        grad_wigner_dt=grad_wigner_dt,
        grad_logits=grad_logits,
        grad_edge=grad_edge,
        grad_z_partial=grad_z_partial,
        grad_z=grad_z,
        grad_focus_src=grad_focus_src,
    )
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        stack.element_type,
        num_bits_per_copy=32,
    )
    channel_thread_layout = cute.make_ordered_layout((1, 32), order=(1, 0))
    reduced_value_layout = cute.make_ordered_layout((10, 1), order=(1, 0))
    raw_tiled_copy = cute.make_tiled_copy_tv(
        copy_atom,
        channel_thread_layout,
        reduced_value_layout,
    )
    raw_layout = cute.make_layout((REDUCED_COUNT, FOCUS_CHANNELS), stride=(32, 1))
    node_count, _, _ = grad_out.shape
    neo_phase_c_backward_layout_kernel(
        params,
        raw_layout,
        raw_tiled_copy,
        focus_eps,
        focus_tau,
        focus_label_smoothing,
    ).launch(
        grid=[node_count, 1, 1],
        block=[HIDDEN, 1, 1],
        stream=stream,
    )
    neo_phase_c_backward_z_reduce_kernel(params, 128).launch(
        grid=[N_FOCUS, 1, 1],
        block=[128, 1, 1],
        stream=stream,
    )


def compile_neo_phase_c_backward_layout(
    *,
    focus_eps: float,
    focus_tau: float,
    focus_label_smoothing: float,
) -> Callable:
    """Compile the exact Neo Phase-C layout-boundary backward callable."""
    if focus_eps <= 0.0:
        raise ValueError("focus_eps must be positive")
    if focus_tau <= 0.0:
        raise ValueError("focus_tau must be positive")
    if not 0.0 <= focus_label_smoothing < 1.0:
        raise ValueError("focus_label_smoothing must be in [0, 1)")

    edge_count = cute.sym_int64()
    node_count = cute.sym_int64()
    fake_grad_out = make_fake_compact_tensor(
        cutlass.Float32,
        (node_count, DEGREE_COUNT, HIDDEN),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_stack = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, N_FOCUS, REDUCED_COUNT, FOCUS_CHANNELS),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_wigner_dt = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, PACKED_WIGNER_VALUES),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_alpha = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, N_FOCUS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_focus_alpha = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, N_FOCUS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_dst_ptr = make_fake_compact_tensor(
        cutlass.Int32,
        (cute.sym_int64(),),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_rotate = make_fake_compact_tensor(
        cutlass.Float32,
        (DEGREE_COUNT,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_edge_gate = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_z_bias = make_fake_compact_tensor(
        cutlass.Float32,
        (N_FOCUS,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_group_max = make_fake_compact_tensor(
        cutlass.Float32,
        (node_count, N_FOCUS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_denom = make_fake_compact_tensor(
        cutlass.Float32,
        (node_count, N_FOCUS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_focus_src = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, N_FOCUS, FOCUS_CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_focus_weight = make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_CHANNELS, N_FOCUS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_focus_scale = make_fake_compact_tensor(
        cutlass.Float32,
        (N_FOCUS, FOCUS_CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_stack = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, N_FOCUS, REDUCED_COUNT, FOCUS_CHANNELS),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_wigner_dt = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, PACKED_WIGNER_VALUES),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_logits = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, N_FOCUS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_edge = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_grad_z_partial = make_fake_compact_tensor(
        cutlass.Float32,
        (node_count, N_FOCUS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_z = make_fake_compact_tensor(
        cutlass.Float32,
        (N_FOCUS,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_grad_focus_src = make_fake_compact_tensor(
        cutlass.Float32,
        (N_FOCUS, edge_count, FOCUS_CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        neo_phase_c_backward_layout_jit,
        fake_grad_out,
        fake_stack,
        fake_wigner_dt,
        fake_alpha,
        fake_focus_alpha,
        fake_dst_ptr,
        fake_rotate,
        fake_edge_gate,
        fake_z_bias,
        fake_group_max,
        fake_denom,
        fake_focus_src,
        fake_focus_weight,
        fake_focus_scale,
        fake_grad_stack,
        fake_grad_wigner_dt,
        fake_grad_logits,
        fake_grad_edge,
        fake_grad_z_partial,
        fake_grad_z,
        fake_grad_focus_src,
        fake_stream,
        focus_eps,
        focus_tau,
        focus_label_smoothing,
        options="--enable-tvm-ffi",
    )
