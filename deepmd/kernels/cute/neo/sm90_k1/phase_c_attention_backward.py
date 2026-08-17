# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Grouped SM90 final Phase-C/attention adjoint with split-state gather.

Four independent 64-thread groups process four CSR edges
per round while sharing one destination node's reverse-linear ``b0/b1``
panels. Each ``(edge, focus)`` warp also retains the expanded
``grad_m0`` and complex ``grad_m1`` accumulators in registers. The same panel
loads therefore serve both the Phase-C scale contraction and split-state
adjoint, replacing the separate ``_expanded_adjoint_gather`` launch.

The focus competition, segmented attention softmax, envelope, and Q/K
adjoints remain fused. No edge-sized temporary is added.
Source-K is atomically accumulated, so callers must clear ``grad_k_node``
before every invocation.
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

from ..compile_cache import (
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
DEGREE_COUNT = 16
M0_WIDTH = 128
M1_WIDTH = 96
PACKED_WIGNER_VALUES = 46
MAX_EDGES_PER_NODE = 128
QK_SCALE = CHANNELS**-0.5
B0_NODE_VALUES = FOCUS_COUNT * DEGREE_COUNT * M0_WIDTH
B1_NODE_VALUES = FOCUS_COUNT * (DEGREE_COUNT - 1) * M1_WIDTH * 2
PACKED_M0 = (0, 1, 2, 3, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31)
PACKED_RE = (4, 5, 6, 15, 16, 17, 18, 19, 32, 33, 34, 35, 36, 37, 38)
PACKED_IM = (7, 8, 9, 20, 21, 22, 23, 24, 39, 40, 41, 42, 43, 44, 45)
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}

EDGE_GROUPS = 4
GROUP_WIDTH = FOCUS_COUNT * CHANNELS
THREADS = EDGE_GROUPS * GROUP_WIDTH
M0_ROWS = M0_WIDTH // CHANNELS
M1_ROWS = M1_WIDTH // CHANNELS


@cute.jit
def _warp_sum(value):
    return cute.arch.warp_reduction_sum(value)


def _fake(dtype, shape: tuple[object, ...], stride_order: tuple[int, ...]):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        **FAKE_TENSOR_KW,
    )


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if (
        tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tensor.data_ptr() % 16
    ):
        raise ValueError(
            f"{name} must be contiguous, 16-byte-aligned {dtype} on {device}"
        )


__all__ = [
    "GroupedExpandedFinalPhaseCAttentionAdjointOutputs",
    "allocate_grouped_expanded_final_phase_c_attention_adjoint_outputs",
    "compile_grouped_expanded_final_phase_c_attention_adjoint",
    "run_grouped_expanded_final_phase_c_attention_adjoint",
]


@dataclass(frozen=True)
class GroupedExpandedFinalPhaseCAttentionAdjointOutputs:
    """True adjoints emitted by the one-launch boundary."""

    grad_m0: torch.Tensor
    grad_m1: torch.Tensor
    grad_dt: torch.Tensor
    grad_logits: torch.Tensor
    grad_edge: torch.Tensor
    grad_focus_src: torch.Tensor
    grad_q_node: torch.Tensor
    grad_k_node: torch.Tensor


@cute.jit
def _grouped_expanded_adjoint_jit(
    b0: cute.Tensor,
    b1_ri: cute.Tensor,
    m0: cute.Tensor,
    m1_ri: cute.Tensor,
    dt_packed: cute.Tensor,
    beta: cute.Tensor,
    alpha: cute.Tensor,
    focus_alpha: cute.Tensor,
    focus_src: cute.Tensor,
    focus_weight: cute.Tensor,
    focus_scale: cute.Tensor,
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    edge_gate: cute.Tensor,
    src: cute.Tensor,
    dst_ptr: cute.Tensor,
    grad_m0: cute.Tensor,
    grad_m1_ri: cute.Tensor,
    grad_dt: cute.Tensor,
    grad_logits: cute.Tensor,
    grad_edge: cute.Tensor,
    grad_focus_src: cute.Tensor,
    grad_q_node: cute.Tensor,
    grad_k_node: cute.Tensor,
    focus_eps: cutlass.Constexpr[float],
    focus_tau: cutlass.Constexpr[float],
    label_smoothing: cutlass.Constexpr[float],
    qk_scale: cutlass.Constexpr[float],
    stream: CUstream,
):
    _grouped_expanded_adjoint_kernel(
        b0,
        b1_ri,
        m0,
        m1_ri,
        dt_packed,
        beta,
        alpha,
        focus_alpha,
        focus_src,
        focus_weight,
        focus_scale,
        q_node,
        k_node,
        edge_gate,
        src,
        dst_ptr,
        grad_m0,
        grad_m1_ri,
        grad_dt,
        grad_logits,
        grad_edge,
        grad_focus_src,
        grad_q_node,
        grad_k_node,
        focus_eps,
        focus_tau,
        label_smoothing,
        qk_scale,
    ).launch(
        grid=[dst_ptr.shape[0] - 1, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _grouped_expanded_adjoint_kernel(
    b0: cute.Tensor,
    b1_ri: cute.Tensor,
    m0: cute.Tensor,
    m1_ri: cute.Tensor,
    dt_packed: cute.Tensor,
    beta: cute.Tensor,
    alpha: cute.Tensor,
    focus_alpha: cute.Tensor,
    focus_src: cute.Tensor,
    focus_weight: cute.Tensor,
    focus_scale: cute.Tensor,
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    edge_gate: cute.Tensor,
    src: cute.Tensor,
    dst_ptr: cute.Tensor,
    grad_m0: cute.Tensor,
    grad_m1_ri: cute.Tensor,
    grad_dt: cute.Tensor,
    grad_logits: cute.Tensor,
    grad_edge: cute.Tensor,
    grad_focus_src: cute.Tensor,
    grad_q_node: cute.Tensor,
    grad_k_node: cute.Tensor,
    focus_eps: cutlass.Constexpr[float],
    focus_tau: cutlass.Constexpr[float],
    label_smoothing: cutlass.Constexpr[float],
    qk_scale: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    node, _, _ = cute.arch.block_idx()
    group = tidx // GROUP_WIDTH
    local = tidx - group * GROUP_WIDTH
    focus = local // CHANNELS
    lane = local - focus * CHANNELS
    node_lo = dst_ptr[node]
    node_hi = dst_ptr[node + 1]
    degree = node_hi - node_lo
    rounds = (degree + EDGE_GROUPS - 1) // EDGE_GROUPS

    smem = cutlass.utils.SmemAllocator()
    b0_node = smem.allocate_tensor(cutlass.Float32, B0_NODE_VALUES)
    b1_node = smem.allocate_tensor(cutlass.Float32, B1_NODE_VALUES)
    beta_adjoint = smem.allocate_tensor(
        cutlass.Float32,
        MAX_EDGES_PER_NODE * FOCUS_COUNT,
    )
    dt_partial = smem.allocate_tensor(
        cutlass.Float32,
        EDGE_GROUPS * FOCUS_COUNT * PACKED_WIGNER_VALUES,
    )
    softmax_dot = smem.allocate_tensor(cutlass.Float32, FOCUS_COUNT)
    q_partial = smem.allocate_tensor(
        cutlass.Float32,
        EDGE_GROUPS * FOCUS_COUNT * CHANNELS,
    )

    for linear in cutlass.range(tidx, B0_NODE_VALUES, THREADS, unroll=1):
        quotient = linear // M0_WIDTH
        feature = linear - quotient * M0_WIDTH
        panel_focus = quotient // DEGREE_COUNT
        q = quotient - panel_focus * DEGREE_COUNT
        b0_node[linear] = b0[panel_focus, q, node, feature].to(cutlass.Float32)
    for linear in cutlass.range(tidx, B1_NODE_VALUES, THREADS, unroll=1):
        quotient = linear // 2
        component = linear - quotient * 2
        feature_quotient = quotient // M1_WIDTH
        feature = quotient - feature_quotient * M1_WIDTH
        panel_focus = feature_quotient // (DEGREE_COUNT - 1)
        q1 = feature_quotient - panel_focus * (DEGREE_COUNT - 1)
        b1_node[linear] = b1_ri[
            panel_focus,
            q1,
            node,
            feature,
            component,
        ].to(cutlass.Float32)
    cute.arch.sync_threads()

    m0_fragment = cute.make_rmem_tensor(
        cute.make_layout((M0_ROWS,), stride=(1,)),
        cutlass.Float32,
    )
    m1_real_fragment = cute.make_rmem_tensor(
        cute.make_layout((M1_ROWS,), stride=(1,)),
        cutlass.Float32,
    )
    m1_imag_fragment = cute.make_rmem_tensor(
        cute.make_layout((M1_ROWS,), stride=(1,)),
        cutlass.Float32,
    )
    grad_m0_fragment = cute.make_rmem_tensor(
        cute.make_layout((M0_ROWS,), stride=(1,)),
        cutlass.Float32,
    )
    grad_m1_real_fragment = cute.make_rmem_tensor(
        cute.make_layout((M1_ROWS,), stride=(1,)),
        cutlass.Float32,
    )
    grad_m1_imag_fragment = cute.make_rmem_tensor(
        cute.make_layout((M1_ROWS,), stride=(1,)),
        cutlass.Float32,
    )

    # All four groups execute the same round count, making CTA barriers safe.
    for edge_round in cutlass.range(rounds, unroll=1):
        edge_slot = edge_round * EDGE_GROUPS + group
        edge = node_lo + edge_slot
        edge_active = edge_slot < degree

        for row in cutlass.range_constexpr(M0_ROWS):
            feature = row * CHANNELS + lane
            value = cutlass.Float32(0.0)
            if edge_active:
                value = m0[focus, edge, feature].to(cutlass.Float32)
            m0_fragment[row] = value
            grad_m0_fragment[row] = cutlass.Float32(0.0)
        for row in cutlass.range_constexpr(M1_ROWS):
            feature = row * CHANNELS + lane
            real = cutlass.Float32(0.0)
            imag = cutlass.Float32(0.0)
            if edge_active:
                real = m1_ri[focus, edge, feature, 0].to(cutlass.Float32)
                imag = m1_ri[focus, edge, feature, 1].to(cutlass.Float32)
            m1_real_fragment[row] = real
            m1_imag_fragment[row] = imag
            grad_m1_real_fragment[row] = cutlass.Float32(0.0)
            grad_m1_imag_fragment[row] = cutlass.Float32(0.0)

        beta_value = cutlass.Float32(0.0)
        if edge_active:
            beta_value = beta[edge, focus].to(cutlass.Float32)
        grad_beta_value = cutlass.Float32(0.0)
        dt_base = (group * FOCUS_COUNT + focus) * PACKED_WIGNER_VALUES

        # Ascending q order matches the existing expanded gather exactly.
        for q in cutlass.range_constexpr(DEGREE_COUNT):
            panel0 = PACKED_M0[q]
            dt0 = cutlass.Float32(0.0)
            if edge_active:
                dt0 = dt_packed[edge, panel0].to(cutlass.Float32)
            scalar0_lane = cutlass.Float32(0.0)
            for row in cutlass.range_constexpr(M0_ROWS):
                feature = row * CHANNELS + lane
                b0_offset = (focus * DEGREE_COUNT + q) * M0_WIDTH + feature
                b_value = b0_node[b0_offset]
                grad_m0_fragment[row] += beta_value * dt0 * b_value
                scalar0_lane += b_value * m0_fragment[row]
            scalar0 = _warp_sum(scalar0_lane)
            if lane == 0:
                dt_partial[dt_base + panel0] = beta_value * scalar0
                grad_beta_value += dt0 * scalar0

            if cutlass.const_expr(q > 0):
                panel_re = PACKED_RE[q - 1]
                panel_im = PACKED_IM[q - 1]
                dt_re = cutlass.Float32(0.0)
                dt_im = cutlass.Float32(0.0)
                if edge_active:
                    dt_re = dt_packed[edge, panel_re].to(cutlass.Float32)
                    dt_im = dt_packed[edge, panel_im].to(cutlass.Float32)
                scalar1_re_lane = cutlass.Float32(0.0)
                scalar1_im_lane = cutlass.Float32(0.0)
                for row in cutlass.range_constexpr(M1_ROWS):
                    feature = row * CHANNELS + lane
                    b1_offset = (
                        (focus * (DEGREE_COUNT - 1) + q - 1) * M1_WIDTH + feature
                    ) * 2
                    br = b1_node[b1_offset]
                    bi = b1_node[b1_offset + 1]
                    grad_m1_real_fragment[row] += beta_value * (dt_re * br - dt_im * bi)
                    grad_m1_imag_fragment[row] += beta_value * (dt_re * bi + dt_im * br)
                    xr = m1_real_fragment[row]
                    xi = m1_imag_fragment[row]
                    scalar1_re_lane += br * xr + bi * xi
                    scalar1_im_lane += br * xi - bi * xr
                scalar1_re = _warp_sum(scalar1_re_lane)
                scalar1_im = _warp_sum(scalar1_im_lane)
                if lane == 0:
                    dt_partial[dt_base + panel_re] = beta_value * scalar1_re
                    dt_partial[dt_base + panel_im] = beta_value * scalar1_im
                    grad_beta_value += dt_re * scalar1_re + dt_im * scalar1_im

        if edge_active:
            for row in cutlass.range_constexpr(M0_ROWS):
                feature = row * CHANNELS + lane
                grad_m0[focus, edge, feature] = grad_m0_fragment[row]
            for row in cutlass.range_constexpr(M1_ROWS):
                feature = row * CHANNELS + lane
                grad_m1_ri[focus, edge, feature, 0] = grad_m1_real_fragment[row]
                grad_m1_ri[focus, edge, feature, 1] = grad_m1_imag_fragment[row]
        if lane == 0:
            beta_adjoint[edge_slot * FOCUS_COUNT + focus] = grad_beta_value
        cute.arch.sync_threads()

        if tidx < EDGE_GROUPS * PACKED_WIGNER_VALUES:
            output_group = tidx // PACKED_WIGNER_VALUES
            panel = tidx - output_group * PACKED_WIGNER_VALUES
            output_slot = edge_round * EDGE_GROUPS + output_group
            if output_slot < degree:
                output_edge = node_lo + output_slot
                focus0 = output_group * FOCUS_COUNT * PACKED_WIGNER_VALUES
                grad_dt[output_edge, panel] = (
                    dt_partial[focus0 + panel]
                    + dt_partial[focus0 + PACKED_WIGNER_VALUES + panel]
                )
        cute.arch.sync_threads()

    # Match the forward path's ascending-edge segmented-softmax reduction order.
    if group == 0 and lane == 0:
        value = cutlass.Float32(0.0)
        for edge_slot in cutlass.range(degree, unroll=1):
            edge = node_lo + edge_slot
            grad_attention = beta_adjoint[
                edge_slot * FOCUS_COUNT + focus
            ] * focus_alpha[edge, focus].to(cutlass.Float32)
            value += alpha[edge, focus].to(cutlass.Float32) * grad_attention
        softmax_dot[focus] = value
    cute.arch.sync_threads()

    keep = cutlass.Float32(1.0 - label_smoothing)
    smooth = cutlass.Float32(label_smoothing / FOCUS_COUNT)
    inv_tau = cutlass.Float32(1.0 / focus_tau)
    grad_q = cutlass.Float32(0.0)

    for edge_slot in cutlass.range(group, degree, EDGE_GROUPS, unroll=1):
        edge = node_lo + edge_slot
        source = src[edge]
        grad_beta_value = beta_adjoint[edge_slot * FOCUS_COUNT + focus]
        alpha_value = alpha[edge, focus].to(cutlass.Float32)
        focus_value = focus_alpha[edge, focus].to(cutlass.Float32)
        grad_attention = grad_beta_value * focus_value
        grad_logit = alpha_value * (grad_attention - softmax_dot[focus])
        if lane == 0:
            grad_logits[edge, focus] = grad_logit

        scaled_grad_logit = grad_logit * cutlass.Float32(qk_scale)
        grad_q += scaled_grad_logit * k_node[source, focus, lane].to(cutlass.Float32)
        grad_k = scaled_grad_logit * q_node[node, focus, lane].to(cutlass.Float32)
        k_offset = (source * FOCUS_COUNT + focus) * CHANNELS + lane
        k_ptr = grad_k_node.iterator + k_offset
        cute.arch.atomic_add(
            k_ptr.llvm_ptr,
            grad_k,
            sem="relaxed",
            scope="gpu",
        )

        probability0 = (focus_alpha[edge, 0].to(cutlass.Float32) - smooth) / keep
        probability1 = (focus_alpha[edge, 1].to(cutlass.Float32) - smooth) / keep
        focus_grad0 = (
            beta_adjoint[edge_slot * FOCUS_COUNT]
            * alpha[edge, 0].to(cutlass.Float32)
            * keep
        )
        focus_grad1 = (
            beta_adjoint[edge_slot * FOCUS_COUNT + 1]
            * alpha[edge, 1].to(cutlass.Float32)
            * keep
        )
        focus_dot = focus_grad0 * probability0 + focus_grad1 * probability1
        probability = probability0
        focus_grad_probability = focus_grad0
        if focus == 1:
            probability = probability1
            focus_grad_probability = focus_grad1
        focus_grad_logit = probability * (focus_grad_probability - focus_dot) * inv_tau

        focus_value_raw = focus_src[edge, focus, lane].to(cutlass.Float32)
        scale = focus_scale[focus, lane].to(cutlass.Float32)
        weight = focus_weight[lane, focus].to(cutlass.Float32)
        inv_rms = cute.rsqrt(
            _warp_sum(focus_value_raw * focus_value_raw) / cutlass.Float32(CHANNELS)
            + cutlass.Float32(focus_eps)
        )
        grad_scaled = focus_grad_logit * weight * scale
        coeff = _warp_sum(grad_scaled * focus_value_raw) / cutlass.Float32(CHANNELS)
        grad_focus_src[focus, edge, lane] = (
            grad_scaled * inv_rms
            - focus_value_raw * inv_rms * inv_rms * inv_rms * coeff
        )

        if local == 0:
            grad_attention0 = beta_adjoint[edge_slot * FOCUS_COUNT] * focus_alpha[
                edge, 0
            ].to(cutlass.Float32)
            grad_attention1 = beta_adjoint[edge_slot * FOCUS_COUNT + 1] * focus_alpha[
                edge, 1
            ].to(cutlass.Float32)
            grad_logit0 = alpha[edge, 0].to(cutlass.Float32) * (
                grad_attention0 - softmax_dot[0]
            )
            grad_logit1 = alpha[edge, 1].to(cutlass.Float32) * (
                grad_attention1 - softmax_dot[1]
            )
            gate = edge_gate[edge].to(cutlass.Float32)
            value = cutlass.Float32(0.0)
            if gate > cutlass.Float32(0.0):
                value = cutlass.Float32(2.0) * (grad_logit0 + grad_logit1) / gate
            grad_edge[edge] = value

    q_partial[group * GROUP_WIDTH + local] = grad_q
    cute.arch.sync_threads()
    if group == 0:
        total = cutlass.Float32(0.0)
        for partial_group in cutlass.range_constexpr(EDGE_GROUPS):
            total += q_partial[partial_group * GROUP_WIDTH + local]
        grad_q_node[node, focus, lane] = total


@device_aware_lru_cache(maxsize=8)
def compile_grouped_expanded_final_phase_c_attention_adjoint(
    focus_eps: float,
    focus_tau: float,
    label_smoothing: float,
    qk_scale: float = QK_SCALE,
) -> Callable:
    """Compile the fixed four-group SM90 schedule."""
    edges = cute.sym_int64()
    nodes = cute.sym_int64()
    b0 = _fake(
        cutlass.Float32,
        (FOCUS_COUNT, DEGREE_COUNT, nodes, M0_WIDTH),
        (3, 2, 1, 0),
    )
    b1 = _fake(
        cutlass.Float32,
        (FOCUS_COUNT, DEGREE_COUNT - 1, nodes, M1_WIDTH, 2),
        (4, 3, 2, 1, 0),
    )
    m0 = _fake(cutlass.Float32, (FOCUS_COUNT, edges, M0_WIDTH), (2, 1, 0))
    m1 = _fake(
        cutlass.Float32,
        (FOCUS_COUNT, edges, M1_WIDTH, 2),
        (3, 2, 1, 0),
    )
    dt = _fake(cutlass.Float32, (edges, PACKED_WIGNER_VALUES), (1, 0))
    edge_focus = _fake(cutlass.Float32, (edges, FOCUS_COUNT), (1, 0))
    focus_src = _fake(
        cutlass.Float32,
        (edges, FOCUS_COUNT, CHANNELS),
        (2, 1, 0),
    )
    focus_weight = _fake(cutlass.Float32, (CHANNELS, FOCUS_COUNT), (1, 0))
    focus_scale = _fake(cutlass.Float32, (FOCUS_COUNT, CHANNELS), (1, 0))
    node_focus = _fake(
        cutlass.Float32,
        (nodes, FOCUS_COUNT, CHANNELS),
        (2, 1, 0),
    )
    edge_scalar = _fake(cutlass.Float32, (edges,), (0,))
    edge_index = _fake(cutlass.Int32, (edges,), (0,))
    dst_ptr = _fake(cutlass.Int32, (cute.sym_int64(),), (0,))
    grad_focus_src = _fake(
        cutlass.Float32,
        (FOCUS_COUNT, edges, CHANNELS),
        (2, 1, 0),
    )
    return cute.compile(
        _grouped_expanded_adjoint_jit,
        b0,
        b1,
        m0,
        m1,
        dt,
        edge_focus,
        edge_focus,
        edge_focus,
        focus_src,
        focus_weight,
        focus_scale,
        node_focus,
        node_focus,
        edge_scalar,
        edge_index,
        dst_ptr,
        _fake(cutlass.Float32, (FOCUS_COUNT, edges, M0_WIDTH), (2, 1, 0)),
        _fake(
            cutlass.Float32,
            (FOCUS_COUNT, edges, M1_WIDTH, 2),
            (3, 2, 1, 0),
        ),
        _fake(cutlass.Float32, (edges, PACKED_WIGNER_VALUES), (1, 0)),
        _fake(cutlass.Float32, (edges, FOCUS_COUNT), (1, 0)),
        _fake(cutlass.Float32, (edges,), (0,)),
        grad_focus_src,
        _fake(cutlass.Float32, (nodes, FOCUS_COUNT, CHANNELS), (2, 1, 0)),
        _fake(cutlass.Float32, (nodes, FOCUS_COUNT, CHANNELS), (2, 1, 0)),
        float(focus_eps),
        float(focus_tau),
        float(label_smoothing),
        float(qk_scale),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def allocate_grouped_expanded_final_phase_c_attention_adjoint_outputs(
    *,
    edge_count: int,
    node_count: int,
    device: torch.device,
    grad_m0: torch.Tensor | None = None,
    grad_m1: torch.Tensor | None = None,
) -> GroupedExpandedFinalPhaseCAttentionAdjointOutputs:
    """Allocate outputs, optionally reusing dead split-state storage."""
    if (grad_m0 is None) != (grad_m1 is None):
        raise ValueError("grad_m0 and grad_m1 must be supplied together")
    opts = {"device": device, "dtype": torch.float32}
    return GroupedExpandedFinalPhaseCAttentionAdjointOutputs(
        grad_m0=(
            grad_m0
            if grad_m0 is not None
            else torch.empty(
                (FOCUS_COUNT, edge_count, M0_WIDTH),
                device=opts["device"],
                dtype=opts["dtype"],
            )
        ),
        grad_m1=(
            grad_m1
            if grad_m1 is not None
            else torch.empty(
                (FOCUS_COUNT, edge_count, M1_WIDTH),
                device=device,
                dtype=torch.complex64,
            )
        ),
        grad_dt=torch.empty(
            (edge_count, PACKED_WIGNER_VALUES),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_logits=torch.empty(
            (edge_count, FOCUS_COUNT),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_edge=torch.empty(
            (edge_count,),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_focus_src=torch.empty(
            (FOCUS_COUNT, edge_count, CHANNELS),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_q_node=torch.empty(
            (node_count, FOCUS_COUNT, CHANNELS),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_k_node=torch.zeros(
            (node_count, FOCUS_COUNT, CHANNELS),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
    )


def run_grouped_expanded_final_phase_c_attention_adjoint(
    *,
    b0: torch.Tensor,
    b1: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
    dt_packed: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    focus_alpha: torch.Tensor,
    focus_src: torch.Tensor,
    focus_weight: torch.Tensor,
    focus_scale: torch.Tensor,
    q_node: torch.Tensor,
    k_node: torch.Tensor,
    edge_gate: torch.Tensor,
    src: torch.Tensor,
    dst_ptr: torch.Tensor,
    focus_eps: float,
    focus_tau: float,
    label_smoothing: float,
    max_degree: int,
    qk_scale: float = QK_SCALE,
    outputs: GroupedExpandedFinalPhaseCAttentionAdjointOutputs | None = None,
) -> GroupedExpandedFinalPhaseCAttentionAdjointOutputs:
    """Run the adjoint; reusable outputs require clearing ``grad_k_node``."""
    device = b0.device
    if device.type != "cuda" or tuple(torch.cuda.get_device_capability(device)) != (
        9,
        0,
    ):
        raise RuntimeError("grouped expanded Phase-C adjoint requires SM90")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict FP32 requires float32 matmul precision 'highest'")
    if max_degree < 0 or max_degree > MAX_EDGES_PER_NODE:
        raise ValueError(
            f"max_degree must be in [0,{MAX_EDGES_PER_NODE}], got {max_degree}"
        )

    node_count = int(dst_ptr.numel() - 1)
    edge_count = int(src.numel())
    expected_inputs = (
        ("b0", b0, (FOCUS_COUNT, DEGREE_COUNT, node_count, M0_WIDTH), torch.float32),
        (
            "b1",
            b1,
            (FOCUS_COUNT, DEGREE_COUNT - 1, node_count, M1_WIDTH),
            torch.complex64,
        ),
        ("m0", m0, (FOCUS_COUNT, edge_count, M0_WIDTH), torch.float32),
        ("m1", m1, (FOCUS_COUNT, edge_count, M1_WIDTH), torch.complex64),
        (
            "dt_packed",
            dt_packed,
            (edge_count, PACKED_WIGNER_VALUES),
            torch.float32,
        ),
        ("beta", beta, (edge_count, FOCUS_COUNT), torch.float32),
        ("alpha", alpha, (edge_count, FOCUS_COUNT), torch.float32),
        ("focus_alpha", focus_alpha, (edge_count, FOCUS_COUNT), torch.float32),
        (
            "focus_src",
            focus_src,
            (edge_count, FOCUS_COUNT, CHANNELS),
            torch.float32,
        ),
        (
            "focus_weight",
            focus_weight,
            (CHANNELS, FOCUS_COUNT),
            torch.float32,
        ),
        (
            "focus_scale",
            focus_scale,
            (FOCUS_COUNT, CHANNELS),
            torch.float32,
        ),
        (
            "q_node",
            q_node,
            (node_count, FOCUS_COUNT, CHANNELS),
            torch.float32,
        ),
        (
            "k_node",
            k_node,
            (node_count, FOCUS_COUNT, CHANNELS),
            torch.float32,
        ),
        ("edge_gate", edge_gate, (edge_count,), torch.float32),
        ("src", src, (edge_count,), torch.int32),
        ("dst_ptr", dst_ptr, (node_count + 1,), torch.int32),
    )
    for name, tensor, shape, dtype in expected_inputs:
        _require_tensor(name, tensor, shape, dtype, device)

    if outputs is None:
        outputs = allocate_grouped_expanded_final_phase_c_attention_adjoint_outputs(
            edge_count=edge_count,
            node_count=node_count,
            device=device,
        )
    expected_outputs = (
        (
            "grad_m0",
            outputs.grad_m0,
            (FOCUS_COUNT, edge_count, M0_WIDTH),
            torch.float32,
        ),
        (
            "grad_m1",
            outputs.grad_m1,
            (FOCUS_COUNT, edge_count, M1_WIDTH),
            torch.complex64,
        ),
        ("grad_dt", outputs.grad_dt, (edge_count, PACKED_WIGNER_VALUES), torch.float32),
        ("grad_logits", outputs.grad_logits, (edge_count, FOCUS_COUNT), torch.float32),
        ("grad_edge", outputs.grad_edge, (edge_count,), torch.float32),
        (
            "grad_focus_src",
            outputs.grad_focus_src,
            (FOCUS_COUNT, edge_count, CHANNELS),
            torch.float32,
        ),
        (
            "grad_q_node",
            outputs.grad_q_node,
            (node_count, FOCUS_COUNT, CHANNELS),
            torch.float32,
        ),
        (
            "grad_k_node",
            outputs.grad_k_node,
            (node_count, FOCUS_COUNT, CHANNELS),
            torch.float32,
        ),
    )
    for name, tensor, shape, dtype in expected_outputs:
        _require_tensor(name, tensor, shape, dtype, device)

    with torch.cuda.device(device):
        compile_grouped_expanded_final_phase_c_attention_adjoint(
            float(focus_eps),
            float(focus_tau),
            float(label_smoothing),
            float(qk_scale),
        )(
            b0,
            torch.view_as_real(b1),
            m0,
            torch.view_as_real(m1),
            dt_packed,
            beta,
            alpha,
            focus_alpha,
            focus_src,
            focus_weight,
            focus_scale,
            q_node,
            k_node,
            edge_gate,
            src,
            dst_ptr,
            outputs.grad_m0,
            torch.view_as_real(outputs.grad_m1),
            outputs.grad_dt,
            outputs.grad_logits,
            outputs.grad_edge,
            outputs.grad_focus_src,
            outputs.grad_q_node,
            outputs.grad_k_node,
        )
    return outputs
