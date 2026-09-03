# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exact split-complex input adjoint for the SM90 Neo Phase-A boundary.

The incoming adjoint remains in the persistent stack's focus-major contract:
``m0`` is float32 ``(2,E,128)`` and ``m1`` is complex64 ``(2,E,96)``.  One
CTA owns one source node and reads those panels directly while recomputing the
packed-Wigner rotation.  It emits the exact input adjoints for node features,
packed Wigner values, compact radial maps, and the rank-1 channel basis without
ever reconstructing an ``(E,2,10,32)`` block-real gradient slab.

Source CSR is a caller-owned edge-cache property.  It preserves physical edge
order while giving each node exclusive ownership of its feature adjoint, so no
atomics or ``(E,16,64)`` edge-local reduction tensor is required.
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
import cutlass.utils as cute_utils
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

from ..compile_cache import (
    device_aware_lru_cache,
)
from ..k1_kernels.cute_neo_radial_phase_a_backward_node import (
    COMPACT_WIDTH,
    DEGREE_COUNT,
    FOCUS_COUNT,
    FOCUS_HIDDEN,
    GROUPS_PER_CTA,
    GROUPS_PER_WARP,
    HIDDEN,
    PACKED_WIGNER_VALUES,
    REDUCED_COUNT,
    SHARED_ROW_PITCH,
    WARP_REDUCTION_GROUP,
    _grad_x_value,
    _recompute_local_value,
    _warp_owned_grad_compact,
    _warp_owned_grad_d,
)
from .persistent import (
    M0_WIDTH,
    M1_WIDTH,
    NeoPersistentComplexState,
    validate_neo_persistent_complex_state,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN202, TC002

THREADS = HIDDEN
REDUCE_THREADS = 32
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}

__all__ = [
    "NeoPhaseAPersistentComplexAdjoints",
    "NeoPhaseAPersistentComplexBackwardWorkspace",
    "allocate_neo_phase_a_persistent_complex_backward",
    "run_neo_phase_a_persistent_complex_backward_fp32",
]


@dataclass(frozen=True)
class NeoPhaseAPersistentComplexAdjoints:
    """Differentiable input adjoints for the strict-FP32 Phase-A producer."""

    grad_x_wide: Tensor
    grad_d_full: Tensor
    grad_radial_compact: Tensor
    grad_channel_basis: Tensor


@dataclass(frozen=True)
class NeoPhaseAPersistentComplexBackwardWorkspace:
    """Small deterministic channel-basis reduction workspace."""

    grad_basis_by_node: Tensor

    @property
    def storage_bytes(self) -> int:
        return self.grad_basis_by_node.numel() * self.grad_basis_by_node.element_size()


@dataclass(frozen=True)
class _BackwardParams:
    grad_m0: cute.Tensor
    grad_m1_ri: cute.Tensor
    radial_compact: cute.Tensor
    channel_basis: cute.Tensor
    x_wide: cute.Tensor
    source_order: cute.Tensor
    source_ptr: cute.Tensor
    d_full: cute.Tensor
    grad_x_wide: cute.Tensor
    grad_d_full: cute.Tensor
    grad_radial_compact: cute.Tensor
    grad_basis_by_node: cute.Tensor


@cute.jit
def _split_grad_value(
    grad_m0,
    grad_m1_ri,
    edge,
    reduced: cutlass.Constexpr[int],
    channel,
):
    focus = channel // FOCUS_HIDDEN
    focus_channel = channel - focus * FOCUS_HIDDEN
    if cutlass.const_expr(reduced < 4):
        return grad_m0[
            focus,
            edge,
            reduced * FOCUS_HIDDEN + focus_channel,
        ].to(cutlass.Float32)
    if cutlass.const_expr(reduced < 7):
        return grad_m1_ri[
            focus,
            edge,
            (reduced - 4) * FOCUS_HIDDEN + focus_channel,
            0,
        ].to(cutlass.Float32)
    return grad_m1_ri[
        focus,
        edge,
        (reduced - 7) * FOCUS_HIDDEN + focus_channel,
        1,
    ].to(cutlass.Float32)


@cute.jit
def _radial_output_value(
    local_values,
    compact,
    reduced: cutlass.Constexpr[int],
    channel,
):
    """Recompute the pre-basis radial output for one reduced row/channel."""
    value = cutlass.Float32(0.0)
    if cutlass.const_expr(reduced < 4):
        for input_row in cutlass.range_constexpr(4):
            value += (
                compact[input_row * 4 + reduced]
                * local_values[input_row * SHARED_ROW_PITCH + channel]
            )
    elif cutlass.const_expr(reduced < 7):
        output_row = reduced - 4
        for input_row in cutlass.range_constexpr(3):
            value += (
                compact[16 + input_row * 3 + output_row]
                * local_values[(4 + input_row) * SHARED_ROW_PITCH + channel]
            )
    else:
        output_row = reduced - 7
        for input_row in cutlass.range_constexpr(3):
            value += (
                compact[16 + input_row * 3 + output_row]
                * local_values[(7 + input_row) * SHARED_ROW_PITCH + channel]
            )
    return value


@cute.jit
def _phase_a_split_backward_jit(
    grad_m0: cute.Tensor,
    grad_m1_ri: cute.Tensor,
    radial_compact: cute.Tensor,
    channel_basis: cute.Tensor,
    x_wide: cute.Tensor,
    source_order: cute.Tensor,
    source_ptr: cute.Tensor,
    d_full: cute.Tensor,
    grad_x_wide: cute.Tensor,
    grad_d_full: cute.Tensor,
    grad_radial_compact: cute.Tensor,
    grad_basis_by_node: cute.Tensor,
    grad_channel_basis: cute.Tensor,
    stream: CUstream,
):
    params = _BackwardParams(
        grad_m0=grad_m0,
        grad_m1_ri=grad_m1_ri,
        radial_compact=radial_compact,
        channel_basis=channel_basis,
        x_wide=x_wide,
        source_order=source_order,
        source_ptr=source_ptr,
        d_full=d_full,
        grad_x_wide=grad_x_wide,
        grad_d_full=grad_d_full,
        grad_radial_compact=grad_radial_compact,
        grad_basis_by_node=grad_basis_by_node,
    )
    _phase_a_split_backward_kernel(params).launch(
        grid=[x_wide.shape[0], 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )
    _reduce_channel_basis_kernel(grad_basis_by_node, grad_channel_basis).launch(
        grid=[HIDDEN, 1, 1],
        block=[REDUCE_THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _phase_a_split_backward_kernel(params: _BackwardParams):
    channel, _, _ = cute.arch.thread_idx()
    node, _, _ = cute.arch.block_idx()
    x_row_pitch = SHARED_ROW_PITCH

    smem = cute_utils.SmemAllocator()
    x_values = smem.allocate_tensor(
        cutlass.Float32,
        DEGREE_COUNT * x_row_pitch,
    )
    focus_grad = smem.allocate_tensor(
        cutlass.Float32,
        REDUCED_COUNT * SHARED_ROW_PITCH,
    )
    # Primal local rows are overwritten by their adjoints after grad-radial and
    # grad-basis have consumed them.
    local_values = smem.allocate_tensor(
        cutlass.Float32,
        REDUCED_COUNT * SHARED_ROW_PITCH,
    )
    d_values = smem.allocate_tensor(cutlass.Float32, PACKED_WIGNER_VALUES)
    compact = smem.allocate_tensor(cutlass.Float32, COMPACT_WIDTH)

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
    grad_basis = cutlass.Float32(0.0)

    lo = params.source_ptr[node]
    hi = params.source_ptr[node + 1]
    for slot in cutlass.range(lo, hi, 1, unroll=1):
        edge = params.source_order[slot]
        for reduced in cutlass.range_constexpr(REDUCED_COUNT):
            focus_grad[reduced * SHARED_ROW_PITCH + channel] = _split_grad_value(
                params.grad_m0,
                params.grad_m1_ri,
                edge,
                reduced,
                channel,
            )
        if channel < COMPACT_WIDTH:
            compact[channel] = params.radial_compact[edge, channel].to(cutlass.Float32)
        if channel < PACKED_WIGNER_VALUES:
            d_values[channel] = params.d_full[edge, channel].to(cutlass.Float32)
        cute.arch.sync_threads()

        _recompute_local_value(
            local_values,
            x_values,
            d_values,
            channel,
            0,
            0,
            0,
            1,
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
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
            SHARED_ROW_PITCH,
            x_row_pitch,
        )
        cute.arch.sync_threads()

        for reduced in cutlass.range_constexpr(REDUCED_COUNT):
            grad_basis += focus_grad[
                reduced * SHARED_ROW_PITCH + channel
            ] * _radial_output_value(
                local_values,
                compact,
                reduced,
                channel,
            )

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
                SHARED_ROW_PITCH,
            )
            if compact_idx < COMPACT_WIDTH:
                if subgroup_lane == 0:
                    params.grad_radial_compact[edge, compact_idx] = grad_value.to(
                        params.grad_radial_compact.element_type
                    )
        # Every warp reads the full shared primal panel while reducing its
        # assigned compact columns.  Do not let an early warp reuse that panel
        # for input adjoints until all compact reductions have finished.
        cute.arch.sync_threads()

        basis = params.channel_basis[channel].to(cutlass.Float32)
        for reduced in cutlass.range_constexpr(REDUCED_COUNT):
            grad_value = cutlass.Float32(0.0)
            if reduced < 4:
                for output_row in cutlass.range_constexpr(4):
                    grad_value += (
                        focus_grad[output_row * SHARED_ROW_PITCH + channel]
                        * compact[reduced * 4 + output_row]
                    )
            elif reduced < 7:
                input_row = reduced - 4
                for output_row in cutlass.range_constexpr(3):
                    grad_value += (
                        focus_grad[(4 + output_row) * SHARED_ROW_PITCH + channel]
                        * compact[16 + input_row * 3 + output_row]
                    )
            else:
                input_row = reduced - 7
                for output_row in cutlass.range_constexpr(3):
                    grad_value += (
                        focus_grad[(7 + output_row) * SHARED_ROW_PITCH + channel]
                        * compact[16 + input_row * 3 + output_row]
                    )
            local_values[reduced * SHARED_ROW_PITCH + channel] = grad_value * basis
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
                SHARED_ROW_PITCH,
                x_row_pitch,
            )
            if panel_idx < PACKED_WIGNER_VALUES:
                if subgroup_lane == 0:
                    params.grad_d_full[edge, panel_idx] = grad_d_value.to(
                        params.grad_d_full.element_type
                    )

        grad_x_0 += _grad_x_value(
            local_values,
            d_values,
            channel,
            0,
            0,
            0,
            1,
            1,
            SHARED_ROW_PITCH,
        )
        grad_x_1 += _grad_x_value(
            local_values,
            d_values,
            channel,
            1,
            0,
            1,
            3,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_2 += _grad_x_value(
            local_values,
            d_values,
            channel,
            1,
            1,
            1,
            3,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_3 += _grad_x_value(
            local_values,
            d_values,
            channel,
            1,
            2,
            1,
            3,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_4 += _grad_x_value(
            local_values,
            d_values,
            channel,
            2,
            0,
            10,
            5,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_5 += _grad_x_value(
            local_values,
            d_values,
            channel,
            2,
            1,
            10,
            5,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_6 += _grad_x_value(
            local_values,
            d_values,
            channel,
            2,
            2,
            10,
            5,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_7 += _grad_x_value(
            local_values,
            d_values,
            channel,
            2,
            3,
            10,
            5,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_8 += _grad_x_value(
            local_values,
            d_values,
            channel,
            2,
            4,
            10,
            5,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_9 += _grad_x_value(
            local_values,
            d_values,
            channel,
            3,
            0,
            25,
            7,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_10 += _grad_x_value(
            local_values,
            d_values,
            channel,
            3,
            1,
            25,
            7,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_11 += _grad_x_value(
            local_values,
            d_values,
            channel,
            3,
            2,
            25,
            7,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_12 += _grad_x_value(
            local_values,
            d_values,
            channel,
            3,
            3,
            25,
            7,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_13 += _grad_x_value(
            local_values,
            d_values,
            channel,
            3,
            4,
            25,
            7,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_14 += _grad_x_value(
            local_values,
            d_values,
            channel,
            3,
            5,
            25,
            7,
            3,
            SHARED_ROW_PITCH,
        )
        grad_x_15 += _grad_x_value(
            local_values,
            d_values,
            channel,
            3,
            6,
            25,
            7,
            3,
            SHARED_ROW_PITCH,
        )
        cute.arch.sync_threads()

    params.grad_x_wide[node, 0 * HIDDEN + channel] = grad_x_0
    params.grad_x_wide[node, 1 * HIDDEN + channel] = grad_x_1
    params.grad_x_wide[node, 2 * HIDDEN + channel] = grad_x_2
    params.grad_x_wide[node, 3 * HIDDEN + channel] = grad_x_3
    params.grad_x_wide[node, 4 * HIDDEN + channel] = grad_x_4
    params.grad_x_wide[node, 5 * HIDDEN + channel] = grad_x_5
    params.grad_x_wide[node, 6 * HIDDEN + channel] = grad_x_6
    params.grad_x_wide[node, 7 * HIDDEN + channel] = grad_x_7
    params.grad_x_wide[node, 8 * HIDDEN + channel] = grad_x_8
    params.grad_x_wide[node, 9 * HIDDEN + channel] = grad_x_9
    params.grad_x_wide[node, 10 * HIDDEN + channel] = grad_x_10
    params.grad_x_wide[node, 11 * HIDDEN + channel] = grad_x_11
    params.grad_x_wide[node, 12 * HIDDEN + channel] = grad_x_12
    params.grad_x_wide[node, 13 * HIDDEN + channel] = grad_x_13
    params.grad_x_wide[node, 14 * HIDDEN + channel] = grad_x_14
    params.grad_x_wide[node, 15 * HIDDEN + channel] = grad_x_15
    params.grad_basis_by_node[node, channel] = grad_basis


@cute.kernel
def _reduce_channel_basis_kernel(
    grad_basis_by_node: cute.Tensor,
    grad_channel_basis: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    channel, _, _ = cute.arch.block_idx()
    value = cutlass.Float32(0.0)
    for node in cutlass.range(
        tidx,
        grad_basis_by_node.shape[0],
        REDUCE_THREADS,
        unroll=1,
    ):
        value += grad_basis_by_node[node, channel].to(cutlass.Float32)
    value = cute.arch.warp_reduction_sum(value)
    if tidx == 0:
        grad_channel_basis[channel] = value


def _fake(dtype, shape: tuple, stride_order: tuple[int, ...]):
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        **FAKE_TENSOR_KW,
    )


@device_aware_lru_cache(maxsize=8)
def _compiled_backward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    if compute_capability != (9, 0):
        raise RuntimeError("the direct split Phase-A adjoint is sm_90-only")
    edge_count = cute.sym_int64()
    node_count = cute.sym_int64()
    source_ptr_count = cute.sym_int64()
    with torch.cuda.device(device_index):
        return cute.compile(
            _phase_a_split_backward_jit,
            _fake(cutlass.Float32, (FOCUS_COUNT, edge_count, M0_WIDTH), (2, 1, 0)),
            _fake(
                cutlass.Float32,
                (FOCUS_COUNT, edge_count, M1_WIDTH, 2),
                (3, 2, 1, 0),
            ),
            _fake(cutlass.Float32, (edge_count, COMPACT_WIDTH), (1, 0)),
            _fake(cutlass.Float32, (HIDDEN,), (0,)),
            _fake(cutlass.Float32, (node_count, DEGREE_COUNT * HIDDEN), (1, 0)),
            _fake(cutlass.Int32, (edge_count,), (0,)),
            _fake(cutlass.Int32, (source_ptr_count,), (0,)),
            _fake(cutlass.Float32, (edge_count, PACKED_WIGNER_VALUES), (1, 0)),
            _fake(cutlass.Float32, (node_count, DEGREE_COUNT * HIDDEN), (1, 0)),
            _fake(cutlass.Float32, (edge_count, PACKED_WIGNER_VALUES), (1, 0)),
            _fake(cutlass.Float32, (edge_count, COMPACT_WIDTH), (1, 0)),
            _fake(cutlass.Float32, (node_count, HIDDEN), (1, 0)),
            _fake(cutlass.Float32, (HIDDEN,), (0,)),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def allocate_neo_phase_a_persistent_complex_backward(
    *,
    edge_count: int,
    node_count: int,
    device: torch.device,
) -> tuple[
    NeoPhaseAPersistentComplexAdjoints,
    NeoPhaseAPersistentComplexBackwardWorkspace,
]:
    """Allocate caller-owned outputs and the bounded basis-reduction workspace."""
    if edge_count <= 0 or node_count <= 0:
        raise ValueError("the Phase-A adjoint requires N > 0 and E > 0")
    opts = {"device": device, "dtype": torch.float32}
    outputs = NeoPhaseAPersistentComplexAdjoints(
        grad_x_wide=torch.empty(
            (node_count, DEGREE_COUNT, HIDDEN),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_d_full=torch.empty(
            (edge_count, PACKED_WIGNER_VALUES),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_radial_compact=torch.empty(
            (edge_count, COMPACT_WIDTH),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
        grad_channel_basis=torch.empty(
            (HIDDEN,),
            device=opts["device"],
            dtype=opts["dtype"],
        ),
    )
    workspace = NeoPhaseAPersistentComplexBackwardWorkspace(
        grad_basis_by_node=torch.empty(
            (node_count, HIDDEN),
            device=opts["device"],
            dtype=opts["dtype"],
        )
    )
    return outputs, workspace


def _expect(
    name: str,
    tensor: Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device or tensor.dtype != dtype or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtype} on {device}")


def run_neo_phase_a_persistent_complex_backward_fp32(
    *,
    grad_state: NeoPersistentComplexState,
    radial_compact: Tensor,
    channel_basis: Tensor,
    x_wide: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    d_full: Tensor,
    outputs: NeoPhaseAPersistentComplexAdjoints | None = None,
    workspace: NeoPhaseAPersistentComplexBackwardWorkspace | None = None,
) -> NeoPhaseAPersistentComplexAdjoints:
    """Run the direct split-gradient Phase-A input adjoint."""
    if torch.is_grad_enabled():
        raise RuntimeError("the explicit Phase-A adjoint must run under no_grad")
    validate_neo_persistent_complex_state(grad_state, name="grad_state")
    device = grad_state.m0.device
    edge_count = grad_state.edge_count
    if x_wide.ndim != 3 or tuple(x_wide.shape[1:]) != (DEGREE_COUNT, HIDDEN):
        raise ValueError("x_wide must have shape (N,16,64)")
    node_count = x_wide.shape[0]
    if node_count <= 0:
        raise ValueError("the Phase-A adjoint requires N > 0")

    specs = (
        ("radial_compact", radial_compact, (edge_count, COMPACT_WIDTH), torch.float32),
        ("channel_basis", channel_basis, (HIDDEN,), torch.float32),
        ("x_wide", x_wide, (node_count, DEGREE_COUNT, HIDDEN), torch.float32),
        ("source_order", source_order, (edge_count,), torch.int32),
        ("source_ptr", source_ptr, (node_count + 1,), torch.int32),
        ("d_full", d_full, (edge_count, PACKED_WIGNER_VALUES), torch.float32),
    )
    for name, tensor, shape, dtype in specs:
        _expect(name, tensor, shape, device, dtype)
    torch._assert_async(
        source_ptr[0] == 0,
        "SM90 Phase-A backward requires source_ptr[0] == 0",
    )
    torch._assert_async(
        source_ptr[-1] == edge_count,
        "SM90 Phase-A backward requires source_ptr[-1] == edge_count",
    )
    if source_ptr.numel() > 1:
        torch._assert_async(
            torch.all(source_ptr[1:] >= source_ptr[:-1]),
            "SM90 Phase-A backward requires nondecreasing source_ptr",
        )
    if source_order.numel() != 0:
        torch._assert_async(
            torch.all((source_order >= 0) & (source_order < edge_count)),
            "SM90 Phase-A backward requires source_order entries in [0, E)",
        )

    if outputs is None or workspace is None:
        if outputs is not None or workspace is not None:
            raise ValueError("outputs and workspace must be supplied together")
        outputs, workspace = allocate_neo_phase_a_persistent_complex_backward(
            edge_count=edge_count,
            node_count=node_count,
            device=device,
        )
    output_specs = (
        ("grad_x_wide", outputs.grad_x_wide, (node_count, DEGREE_COUNT, HIDDEN)),
        ("grad_d_full", outputs.grad_d_full, (edge_count, PACKED_WIGNER_VALUES)),
        (
            "grad_radial_compact",
            outputs.grad_radial_compact,
            (edge_count, COMPACT_WIDTH),
        ),
        ("grad_channel_basis", outputs.grad_channel_basis, (HIDDEN,)),
        (
            "grad_basis_by_node",
            workspace.grad_basis_by_node,
            (node_count, HIDDEN),
        ),
    )
    for name, tensor, shape in output_specs:
        _expect(name, tensor, shape, device, torch.float32)

    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict FP32 requires float32 matmul precision 'highest'")
    device_index = device.index
    if device_index is None:
        raise RuntimeError("the direct split Phase-A adjoint requires CUDA")
    compute_capability = tuple(torch.cuda.get_device_capability(device_index))
    kernel = _compiled_backward(device_index, compute_capability)
    grad_m1_ri = torch.view_as_real(grad_state.m1)
    if not grad_m1_ri.is_contiguous():
        raise ValueError("grad_state.m1 must expose interleaved real/imag storage")
    with torch.cuda.device(device):
        kernel(
            grad_state.m0,
            grad_m1_ri,
            radial_compact,
            channel_basis,
            x_wide.view(node_count, DEGREE_COUNT * HIDDEN),
            source_order,
            source_ptr,
            d_full,
            outputs.grad_x_wide.view(node_count, DEGREE_COUNT * HIDDEN),
            outputs.grad_d_full,
            outputs.grad_radial_compact,
            workspace.grad_basis_by_node,
            outputs.grad_channel_basis,
        )
    return outputs
