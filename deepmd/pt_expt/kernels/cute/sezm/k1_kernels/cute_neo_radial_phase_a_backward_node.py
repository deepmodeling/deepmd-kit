# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Source-CSR node-tiled Neo radial and Phase-A backward.

This is the contention-free node-tiled implementation for the exact Neo K1 shape
``lmax=3, D=16, Dm=10, Cwide=64, F=2``. Edge tensors retain their physical
destination-sorted order. ``source_ptr`` delimits intervals in
``source_order``, whose slots hold the corresponding physical edge ids.

One 64-thread CTA owns one source node. It keeps the node feature row in shared
memory and 16 adjoint values per thread in registers, processes all incident
edges, writes the per-edge radial/Wigner adjoints directly, then writes the
node adjoint once. There are no global atomics and no ``(E, 16, 64)``
reduction intermediate.
"""

# ruff: noqa: ANN001, ANN201, ANN202, TC002, UP035

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
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

from ..k1_wigner_layout import PACKED_VALUE_COUNT as PACKED_WIGNER_VALUES

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}

DEGREE_COUNT = 16
REDUCED_COUNT = 10
HIDDEN = 64
FOCUS_COUNT = 2
FOCUS_HIDDEN = 32
FOCUS_ROW = REDUCED_COUNT * FOCUS_HIDDEN
RADIAL_WIDTH = 4 * FOCUS_HIDDEN
COMPACT_WIDTH = 25
SHARED_ROW_PITCH = HIDDEN + 4
WARP_REDUCTION_GROUP = 4
GROUPS_PER_WARP = 32 // WARP_REDUCTION_GROUP
GROUPS_PER_CTA = 2 * GROUPS_PER_WARP
CHANNELS_PER_SUBGROUP_LANE = HIDDEN // WARP_REDUCTION_GROUP


@dataclass(frozen=True)
class NeoRadialPhaseABackwardNodeParams:
    grad_out_focus: cute.Tensor
    grad_focus_src: cute.Tensor
    grad_logits: cute.Tensor
    radial_state: cute.Tensor
    channel_basis: cute.Tensor
    x_wide: cute.Tensor
    source_order: cute.Tensor
    source_ptr: cute.Tensor
    d_full: cute.Tensor
    grad_x_wide: cute.Tensor
    grad_d_full: cute.Tensor


@cute.jit
def _focus_grad_value(
    grad_out_focus: cute.Tensor,
    grad_focus_src: cute.Tensor,
    edge,
    coeff: cutlass.Constexpr[int],
    channel,
):
    focus = channel // FOCUS_HIDDEN
    focus_channel = channel - focus * FOCUS_HIDDEN
    offset = focus * FOCUS_ROW + coeff * FOCUS_HIDDEN + focus_channel
    value = grad_out_focus[edge, offset].to(cutlass.Float32)
    if cutlass.const_expr(coeff == 0):
        value += grad_focus_src[focus, edge, focus_channel].to(cutlass.Float32)
    return value


@cute.jit
def _recompute_local_value(
    local_values,
    x_values,
    d_values,
    channel,
    reduced: cutlass.Constexpr[int],
    panel_start: cutlass.Constexpr[int],
    full_start: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    shared_row_pitch: cutlass.Constexpr[int],
    x_row_pitch: cutlass.Constexpr[int],
):
    acc = cutlass.Float32(0.0)
    for local_col in cutlass.range_constexpr(width):
        acc += (
            d_values[panel_start + local_col]
            * x_values[(full_start + local_col) * x_row_pitch + channel]
        )
    local_values[reduced * shared_row_pitch + channel] = acc


@cute.jit
def _warp_owned_grad_compact(
    focus_grad,
    local_values,
    channel_basis: cute.Tensor,
    compact_idx,
    subgroup_lane,
    shared_row_pitch: cutlass.Constexpr[int],
):
    """Reduce one compact-kernel gradient inside a four-lane subgroup."""
    value = cutlass.Float32(0.0)
    if compact_idx < 16:
        in_coeff = compact_idx // 4
        out_coeff = compact_idx - in_coeff * 4
        for channel_step in cutlass.range_constexpr(CHANNELS_PER_SUBGROUP_LANE):
            hidden_channel = subgroup_lane + channel_step * WARP_REDUCTION_GROUP
            basis = channel_basis[hidden_channel].to(cutlass.Float32)
            value += (
                focus_grad[out_coeff * shared_row_pitch + hidden_channel]
                * local_values[in_coeff * shared_row_pitch + hidden_channel]
                * basis
            )
    else:
        pair = compact_idx - 16
        in_coeff = pair // 3
        out_coeff = pair - in_coeff * 3
        for channel_step in cutlass.range_constexpr(CHANNELS_PER_SUBGROUP_LANE):
            hidden_channel = subgroup_lane + channel_step * WARP_REDUCTION_GROUP
            basis = channel_basis[hidden_channel].to(cutlass.Float32)
            value += basis * (
                focus_grad[(4 + out_coeff) * shared_row_pitch + hidden_channel]
                * local_values[(4 + in_coeff) * shared_row_pitch + hidden_channel]
                + focus_grad[(7 + out_coeff) * shared_row_pitch + hidden_channel]
                * local_values[(7 + in_coeff) * shared_row_pitch + hidden_channel]
            )
    return cute.arch.warp_reduction_sum(
        value,
        threads_in_group=WARP_REDUCTION_GROUP,
    )


@cute.jit
def _warp_owned_grad_d(
    local_values,
    x_values,
    panel_idx,
    subgroup_lane,
    shared_row_pitch: cutlass.Constexpr[int],
    x_row_pitch: cutlass.Constexpr[int],
):
    """Reduce one packed Wigner adjoint inside a four-lane subgroup."""
    reduced = cutlass.Int32(0)
    full_col = cutlass.Int32(0)
    if panel_idx >= 25:
        local_idx = panel_idx - 25
        row_slot = local_idx // 7
        reduced = 3 + row_slot * 3
        full_col = 9 + local_idx - row_slot * 7
    elif panel_idx >= 10:
        local_idx = panel_idx - 10
        row_slot = local_idx // 5
        reduced = 2 + row_slot * 3
        full_col = 4 + local_idx - row_slot * 5
    elif panel_idx >= 1:
        local_idx = panel_idx - 1
        row_slot = local_idx // 3
        reduced = 1 + row_slot * 3
        full_col = 1 + local_idx - row_slot * 3

    value = cutlass.Float32(0.0)
    for channel_step in cutlass.range_constexpr(CHANNELS_PER_SUBGROUP_LANE):
        hidden_channel = subgroup_lane + channel_step * WARP_REDUCTION_GROUP
        value += (
            local_values[reduced * shared_row_pitch + hidden_channel]
            * x_values[full_col * x_row_pitch + hidden_channel]
        )
    return cute.arch.warp_reduction_sum(
        value,
        threads_in_group=WARP_REDUCTION_GROUP,
    )


@cute.jit
def _grad_x_value(
    local_values,
    d_values,
    channel,
    degree: cutlass.Constexpr[int],
    local_col: cutlass.Constexpr[int],
    panel_start: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    shared_row_pitch: cutlass.Constexpr[int],
):
    acc = cutlass.Float32(0.0)
    for row_slot in cutlass.range_constexpr(rows):
        reduced = degree + row_slot * 3
        panel_offset = panel_start + row_slot * width + local_col
        acc += (
            d_values[panel_offset] * local_values[reduced * shared_row_pitch + channel]
        )
    return acc


@cute.jit
def neo_radial_phase_a_backward_node_jit(
    grad_out_focus: cute.Tensor,
    grad_focus_src: cute.Tensor,
    grad_logits: cute.Tensor,
    radial_state: cute.Tensor,
    channel_basis: cute.Tensor,
    x_wide: cute.Tensor,
    source_order: cute.Tensor,
    source_ptr: cute.Tensor,
    d_full: cute.Tensor,
    grad_x_wide: cute.Tensor,
    grad_d_full: cute.Tensor,
    stream: CUstream,
):
    params = NeoRadialPhaseABackwardNodeParams(
        grad_out_focus=grad_out_focus,
        grad_focus_src=grad_focus_src,
        grad_logits=grad_logits,
        radial_state=radial_state,
        channel_basis=channel_basis,
        x_wide=x_wide,
        source_order=source_order,
        source_ptr=source_ptr,
        d_full=d_full,
        grad_x_wide=grad_x_wide,
        grad_d_full=grad_d_full,
    )
    node_count, _ = grad_x_wide.shape
    neo_radial_phase_a_backward_node_kernel(params).launch(
        grid=[node_count, 1, 1],
        block=[HIDDEN, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_radial_phase_a_backward_node_kernel(
    params: NeoRadialPhaseABackwardNodeParams,
):
    channel, _, _ = cute.arch.thread_idx()
    node, _, _ = cute.arch.block_idx()
    shared_row_pitch = SHARED_ROW_PITCH

    smem = cutlass.utils.SmemAllocator()
    x_row_pitch = SHARED_ROW_PITCH
    x_values = smem.allocate_tensor(
        cutlass.Float32,
        DEGREE_COUNT * x_row_pitch,
    )
    focus_grad = smem.allocate_tensor(
        cutlass.Float32,
        REDUCED_COUNT * SHARED_ROW_PITCH,
    )
    # The primal local rows are dead after grad_compact. Reuse this panel for
    # their adjoints instead of reserving another 2.5 KiB per CTA.
    local_values = smem.allocate_tensor(
        cutlass.Float32,
        REDUCED_COUNT * SHARED_ROW_PITCH,
    )
    d_values = smem.allocate_tensor(cutlass.Float32, PACKED_WIGNER_VALUES)
    compact = smem.allocate_tensor(cutlass.Float32, COMPACT_WIDTH)
    grad_compact = smem.allocate_tensor(cutlass.Float32, COMPACT_WIDTH)

    for full_row in cutlass.range_constexpr(DEGREE_COUNT):
        x_values[full_row * x_row_pitch + channel] = params.x_wide[
            node,
            full_row * HIDDEN + channel,
        ].to(cutlass.Float32)
    cute.arch.sync_threads()

    grad_x_0 = cutlass.Float32(0.0)
    grad_x_1 = cutlass.Float32(0.0)
    grad_x_2 = cutlass.Float32(0.0)
    grad_x_3 = cutlass.Float32(0.0)
    grad_x_4 = cutlass.Float32(0.0)
    grad_x_5 = cutlass.Float32(0.0)
    grad_x_6 = cutlass.Float32(0.0)
    grad_x_7 = cutlass.Float32(0.0)
    grad_x_8 = cutlass.Float32(0.0)
    grad_x_9 = cutlass.Float32(0.0)
    grad_x_10 = cutlass.Float32(0.0)
    grad_x_11 = cutlass.Float32(0.0)
    grad_x_12 = cutlass.Float32(0.0)
    grad_x_13 = cutlass.Float32(0.0)
    grad_x_14 = cutlass.Float32(0.0)
    grad_x_15 = cutlass.Float32(0.0)

    lo = params.source_ptr[node]
    hi = params.source_ptr[node + 1]
    for slot in cutlass.range(lo, hi, 1, unroll=1):
        edge = params.source_order[slot]
        for coeff in cutlass.range_constexpr(REDUCED_COUNT):
            row_offset = coeff * shared_row_pitch + channel
            focus_grad[row_offset] = _focus_grad_value(
                params.grad_out_focus,
                params.grad_focus_src,
                edge,
                coeff,
                channel,
            )
        if channel < COMPACT_WIDTH:
            compact[channel] = params.radial_state[edge, channel].to(cutlass.Float32)

        if channel < PACKED_WIGNER_VALUES:
            d_values[channel] = params.d_full[edge, channel].to(cutlass.Float32)
        cute.arch.sync_threads()

        if cutlass.const_expr(True):
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                0,
                0,
                0,
                1,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                1,
                1,
                1,
                3,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                2,
                10,
                4,
                5,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                3,
                25,
                9,
                7,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                4,
                4,
                1,
                3,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                5,
                15,
                4,
                5,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                6,
                32,
                9,
                7,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                7,
                7,
                1,
                3,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                8,
                20,
                4,
                5,
                shared_row_pitch,
                x_row_pitch,
            )
            _recompute_local_value(
                local_values,
                x_values,
                d_values,
                channel,
                9,
                39,
                9,
                7,
                shared_row_pitch,
                x_row_pitch,
            )
            cute.arch.sync_threads()

        lane = channel % 32
        warp = channel // 32
        subgroup = lane // WARP_REDUCTION_GROUP
        subgroup_lane = lane % WARP_REDUCTION_GROUP
        group = warp * GROUPS_PER_WARP + subgroup
        for batch in cutlass.range_constexpr(
            (COMPACT_WIDTH + GROUPS_PER_CTA - 1) // GROUPS_PER_CTA
        ):
            compact_idx = batch * GROUPS_PER_CTA + group
            safe_compact_idx = compact_idx
            if compact_idx >= COMPACT_WIDTH:
                safe_compact_idx = cutlass.Int32(0)
            grad_value = _warp_owned_grad_compact(
                focus_grad,
                local_values,
                params.channel_basis,
                safe_compact_idx,
                subgroup_lane,
                shared_row_pitch,
            )
            if compact_idx < COMPACT_WIDTH:
                if subgroup_lane == 0:
                    grad_compact[compact_idx] = grad_value
        cute.arch.sync_threads()

        basis = params.channel_basis[channel].to(cutlass.Float32)
        for coeff in cutlass.range_constexpr(REDUCED_COUNT):
            grad_value = cutlass.Float32(0.0)
            if coeff < 4:
                for out_coeff in cutlass.range_constexpr(4):
                    grad_value += (
                        focus_grad[out_coeff * shared_row_pitch + channel]
                        * compact[coeff * 4 + out_coeff]
                    )
            elif coeff < 7:
                in_coeff = coeff - 4
                for out_coeff in cutlass.range_constexpr(3):
                    grad_value += (
                        focus_grad[(4 + out_coeff) * shared_row_pitch + channel]
                        * compact[16 + in_coeff * 3 + out_coeff]
                    )
            else:
                in_coeff = coeff - 7
                for out_coeff in cutlass.range_constexpr(3):
                    grad_value += (
                        focus_grad[(7 + out_coeff) * shared_row_pitch + channel]
                        * compact[16 + in_coeff * 3 + out_coeff]
                    )
            local_values[coeff * shared_row_pitch + channel] = grad_value * basis

        if channel < COMPACT_WIDTH:
            # grad_out_focus is dead once this edge has been reduced. Pack
            # the 27-column GEMM operand in-place to avoid a separate cat.
            params.grad_out_focus[edge, channel] = grad_compact[channel].to(
                params.grad_out_focus.element_type
            )
        if channel < FOCUS_COUNT:
            params.grad_out_focus[edge, COMPACT_WIDTH + channel] = params.grad_logits[
                edge, channel
            ].to(params.grad_out_focus.element_type)
        cute.arch.sync_threads()

        for batch in cutlass.range_constexpr(
            (PACKED_WIGNER_VALUES + GROUPS_PER_CTA - 1) // GROUPS_PER_CTA
        ):
            panel_idx = batch * GROUPS_PER_CTA + group
            safe_panel_idx = panel_idx
            if panel_idx >= PACKED_WIGNER_VALUES:
                safe_panel_idx = cutlass.Int32(0)
            grad_d_value = _warp_owned_grad_d(
                local_values,
                x_values,
                safe_panel_idx,
                subgroup_lane,
                shared_row_pitch,
                x_row_pitch,
            )
            if panel_idx < PACKED_WIGNER_VALUES:
                if subgroup_lane == 0:
                    params.grad_d_full[edge, panel_idx] = grad_d_value.to(
                        params.grad_d_full.element_type
                    )

        grad_x_0 += _grad_x_value(
            local_values, d_values, channel, 0, 0, 0, 1, 1, shared_row_pitch
        )
        grad_x_1 += _grad_x_value(
            local_values, d_values, channel, 1, 0, 1, 3, 3, shared_row_pitch
        )
        grad_x_2 += _grad_x_value(
            local_values, d_values, channel, 1, 1, 1, 3, 3, shared_row_pitch
        )
        grad_x_3 += _grad_x_value(
            local_values, d_values, channel, 1, 2, 1, 3, 3, shared_row_pitch
        )
        grad_x_4 += _grad_x_value(
            local_values, d_values, channel, 2, 0, 10, 5, 3, shared_row_pitch
        )
        grad_x_5 += _grad_x_value(
            local_values, d_values, channel, 2, 1, 10, 5, 3, shared_row_pitch
        )
        grad_x_6 += _grad_x_value(
            local_values, d_values, channel, 2, 2, 10, 5, 3, shared_row_pitch
        )
        grad_x_7 += _grad_x_value(
            local_values, d_values, channel, 2, 3, 10, 5, 3, shared_row_pitch
        )
        grad_x_8 += _grad_x_value(
            local_values, d_values, channel, 2, 4, 10, 5, 3, shared_row_pitch
        )
        grad_x_9 += _grad_x_value(
            local_values, d_values, channel, 3, 0, 25, 7, 3, shared_row_pitch
        )
        grad_x_10 += _grad_x_value(
            local_values, d_values, channel, 3, 1, 25, 7, 3, shared_row_pitch
        )
        grad_x_11 += _grad_x_value(
            local_values, d_values, channel, 3, 2, 25, 7, 3, shared_row_pitch
        )
        grad_x_12 += _grad_x_value(
            local_values, d_values, channel, 3, 3, 25, 7, 3, shared_row_pitch
        )
        grad_x_13 += _grad_x_value(
            local_values, d_values, channel, 3, 4, 25, 7, 3, shared_row_pitch
        )
        grad_x_14 += _grad_x_value(
            local_values, d_values, channel, 3, 5, 25, 7, 3, shared_row_pitch
        )
        grad_x_15 += _grad_x_value(
            local_values, d_values, channel, 3, 6, 25, 7, 3, shared_row_pitch
        )
        cute.arch.sync_threads()

    params.grad_x_wide[node, 0 * HIDDEN + channel] = grad_x_0.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 1 * HIDDEN + channel] = grad_x_1.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 2 * HIDDEN + channel] = grad_x_2.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 3 * HIDDEN + channel] = grad_x_3.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 4 * HIDDEN + channel] = grad_x_4.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 5 * HIDDEN + channel] = grad_x_5.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 6 * HIDDEN + channel] = grad_x_6.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 7 * HIDDEN + channel] = grad_x_7.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 8 * HIDDEN + channel] = grad_x_8.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 9 * HIDDEN + channel] = grad_x_9.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 10 * HIDDEN + channel] = grad_x_10.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 11 * HIDDEN + channel] = grad_x_11.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 12 * HIDDEN + channel] = grad_x_12.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 13 * HIDDEN + channel] = grad_x_13.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 14 * HIDDEN + channel] = grad_x_14.to(
        params.grad_x_wide.element_type
    )
    params.grad_x_wide[node, 15 * HIDDEN + channel] = grad_x_15.to(
        params.grad_x_wide.element_type
    )


def compile_neo_radial_phase_a_backward_node_tiled() -> Callable:
    """Compile the exact-shape source-CSR node-tiled backward specialization."""
    edge_count = cute.sym_int64()
    node_count = cute.sym_int64()
    source_ptr_count = cute.sym_int64()
    fake_grad_out = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, FOCUS_COUNT * REDUCED_COUNT * FOCUS_HIDDEN),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_focus_src = make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, edge_count, FOCUS_HIDDEN),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_logits = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, FOCUS_COUNT),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_radial_state = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, COMPACT_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_basis = make_fake_compact_tensor(
        cutlass.Float32,
        (HIDDEN,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_x_wide = make_fake_compact_tensor(
        cutlass.Float32,
        (node_count, DEGREE_COUNT * HIDDEN),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_source_order = make_fake_compact_tensor(
        cutlass.Int32,
        (edge_count,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_source_ptr = make_fake_compact_tensor(
        cutlass.Int32,
        (source_ptr_count,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_d = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, PACKED_WIGNER_VALUES),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_x = make_fake_compact_tensor(
        cutlass.Float32,
        (node_count, DEGREE_COUNT * HIDDEN),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_grad_d = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, PACKED_WIGNER_VALUES),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        neo_radial_phase_a_backward_node_jit,
        fake_grad_out,
        fake_grad_focus_src,
        fake_grad_logits,
        fake_radial_state,
        fake_basis,
        fake_x_wide,
        fake_source_order,
        fake_source_ptr,
        fake_d,
        fake_grad_x,
        fake_grad_d,
        fake_stream,
        options="--enable-tvm-ffi",
    )
