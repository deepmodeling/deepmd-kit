# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fused strict-FP32 Neo message-grid input adjoint for SM90."""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
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

from ...compile_cache import (
    device_aware_lru_cache,
)
from ...runtime_policy import (
    SM90_CAPABILITY,
    uses_strict_fp32_matmul,
)
from .message_grid_gaunt import (
    Sm90GauntSchedule,
    build_sm90_gaunt_schedule,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN202, ANN204, TC002

DEGREE_COUNT = 16
FRAME_COUNT = 3
FOCUS_COUNT = 2
CHANNELS = 32
FOLDED_CHANNELS = FOCUS_COUNT * CHANNELS
PACKED_COEFF_DIM = DEGREE_COUNT * FRAME_COUNT

FRAME_CONTRACT_THREADS = 128
FRAME_CONTRACT_M_TILE = 128
FRAME_CONTRACT_N_TILE = 64
FRAME_CONTRACT_K_TILE = 16
FRAME_CONTRACT_N_TILES = 2
FRAME_CONTRACT_WIDTH = FRAME_COUNT * CHANNELS
FRAME_CONTRACT_SMEM_PADDING = 4

READOUT_THREADS = 256
READOUT_GROUPS = 4
VALUES_PER_PANEL = PACKED_COEFF_DIM * FOLDED_CHANNELS
ROWS_PER_GROUP = PACKED_COEFF_DIM // READOUT_GROUPS
DEGREE_CHANNEL_VALUES_PER_NODE = DEGREE_COUNT * CHANNELS
DEGREES_PER_THREAD = DEGREE_CHANNEL_VALUES_PER_NODE // READOUT_THREADS
LEFT_PANEL = 0
RIGHT_PANEL = 1
GATED_COEFFICIENT_PANEL = 2
GRAD_LEFT_PANEL = 3
GRAD_RIGHT_PANEL = 4
WORKSPACE_PANELS = 5

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@dataclass(frozen=True)
class Sm90MessageGridState:
    """Certified Gaunt schedule and packed immutable readout weights."""

    schedule: Sm90GauntSchedule
    frame_contract: torch.Tensor
    residual_scale: torch.Tensor
    frame_expand_t: torch.Tensor


def _tensor_cache_key(tensor: torch.Tensor | None) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tensor._version,
        tensor.dtype,
        tensor.device,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def prepare_sm90_message_grid_state(net: Any) -> Sm90MessageGridState:
    """Prepare and cache the fixed Neo readout contract outside the hot path."""
    projector = net.projector
    key = (
        _tensor_cache_key(projector.to_grid_mat),
        _tensor_cache_key(projector.from_grid_mat),
        _tensor_cache_key(net.frame_contract.weight),
        _tensor_cache_key(net.frame_contract.degree_index),
        _tensor_cache_key(net.frame_expand.weight),
        _tensor_cache_key(net.frame_expand.degree_index),
        _tensor_cache_key(net.residual_scale),
    )
    cached = getattr(net, "_deepmd_cute_sm90_message_grid_state", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    schedule = build_sm90_gaunt_schedule(
        projector.to_grid_mat,
        projector.from_grid_mat,
    )
    frame_contract = net.frame_contract.weight.index_select(
        0,
        net.frame_contract.degree_index,
    ).view(DEGREE_COUNT, FRAME_COUNT, CHANNELS, CHANNELS)
    frame_expand = net.frame_expand.weight.index_select(
        0,
        net.frame_expand.degree_index,
    ).view(DEGREE_COUNT, CHANNELS, FRAME_COUNT, CHANNELS)
    residual_scale = net.residual_scale
    if residual_scale is None:
        residual_scale = torch.ones(
            (FOCUS_COUNT, CHANNELS),
            device=frame_contract.device,
            dtype=torch.float32,
        )
    state = Sm90MessageGridState(
        schedule=schedule,
        frame_contract=frame_contract.contiguous(),
        residual_scale=residual_scale.view(FOCUS_COUNT, CHANNELS).contiguous(),
        frame_expand_t=frame_expand.permute(0, 2, 3, 1).contiguous(),
    )
    net._deepmd_cute_sm90_message_grid_state = (key, state)
    return state


@cute.jit
def _make_frame_contract_mma():
    atoms_layout = cute.make_layout(
        (FRAME_CONTRACT_THREADS // 16, 16, 1),
        stride=(16, 1, 0),
    )
    permutation_m = cute.make_layout(
        (atoms_layout.shape[0], 4),
        stride=(4, 1),
    )
    permutation_n = cute.make_layout(
        (atoms_layout.shape[1], 4),
        stride=(4, 1),
    )
    return cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(cutlass.Float32),
        atoms_layout,
        permutation_mnk=(permutation_m, permutation_n, None),
    )


class _FrameContractAdjoint:
    """Batched ``N x 32 @ 32 x 96`` strict-FP32 FrameContract adjoint."""

    @cute.jit
    def __call__(
        self,
        grad_out: cute.Tensor,
        frame_contract: cute.Tensor,
        residual_scale: cute.Tensor,
        coefficient_slab: cute.Tensor,
        stream: CUstream,
    ):
        s_a_layout = cute.make_layout(
            (FRAME_CONTRACT_M_TILE, FRAME_CONTRACT_K_TILE),
            stride=(1, FRAME_CONTRACT_M_TILE + FRAME_CONTRACT_SMEM_PADDING),
        )
        s_b_layout = cute.make_layout(
            (FRAME_CONTRACT_N_TILE, FRAME_CONTRACT_K_TILE),
            stride=(1, FRAME_CONTRACT_N_TILE + FRAME_CONTRACT_SMEM_PADDING),
        )
        output_reference_layout = cute.make_layout(
            (FRAME_CONTRACT_M_TILE, FRAME_CONTRACT_N_TILE),
            stride=(FRAME_CONTRACT_N_TILE, 1),
        )
        self.kernel(
            grad_out,
            frame_contract,
            residual_scale,
            coefficient_slab,
            s_a_layout,
            s_b_layout,
            output_reference_layout,
            _make_frame_contract_mma(),
        ).launch(
            grid=(
                cute.ceil_div(grad_out.shape[0], FRAME_CONTRACT_M_TILE),
                FRAME_CONTRACT_N_TILES,
                DEGREE_COUNT * FOCUS_COUNT,
            ),
            block=[FRAME_CONTRACT_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        grad_out: cute.Tensor,
        frame_contract: cute.Tensor,
        residual_scale: cute.Tensor,
        coefficient_slab: cute.Tensor,
        s_a_layout: cute.Layout,
        s_b_layout: cute.Layout,
        output_reference_layout: cute.Layout,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, batch = cute.arch.block_idx()
        m_base = m_tile * FRAME_CONTRACT_M_TILE
        n_base = n_tile * FRAME_CONTRACT_N_TILE
        degree = batch // FOCUS_COUNT
        focus = batch - degree * FOCUS_COUNT

        smem = cute_utils.SmemAllocator()
        s_a = smem.allocate_tensor(cutlass.Float32, s_a_layout, 16)
        s_b = smem.allocate_tensor(cutlass.Float32, s_b_layout, 16)
        thr_mma = tiled_mma.get_slice(tidx)
        t_s_a = thr_mma.partition_A(s_a)
        t_s_b = thr_mma.partition_B(s_b)
        r_a = tiled_mma.make_fragment_A(t_s_a)
        r_b = tiled_mma.make_fragment_B(t_s_b)

        # The reference supplies the C-fragment partition. The epilogue maps
        # logical columns directly into the packed coefficient slab.
        output_reference = cute.make_tensor(
            coefficient_slab.iterator,
            output_reference_layout,
        )
        t_c_reference = thr_mma.partition_C(output_reference)
        accumulator = tiled_mma.make_fragment_C(t_c_reference)
        accumulator.fill(0.0)
        k_blocks = cute.size(r_a, mode=[2])

        for k_tile in cutlass.range_constexpr(CHANNELS // FRAME_CONTRACT_K_TILE):
            self._stage_operands(
                grad_out,
                frame_contract,
                residual_scale,
                s_a,
                s_b,
                tidx,
                m_base,
                n_base,
                degree,
                focus,
                k_tile * FRAME_CONTRACT_K_TILE,
            )
            for k_block in cutlass.range(k_blocks, unroll_full=True):
                cute.autovec_copy(
                    t_s_a[None, None, k_block],
                    r_a[None, None, k_block],
                )
                cute.autovec_copy(
                    t_s_b[None, None, k_block],
                    r_b[None, None, k_block],
                )
                cute.gemm(
                    tiled_mma,
                    accumulator,
                    r_a[None, None, k_block],
                    r_b[None, None, k_block],
                    accumulator,
                )
            cute.arch.sync_threads()

        self._store_compact_epilogue(
            coefficient_slab,
            accumulator,
            thr_mma,
            output_reference,
            m_base,
            n_base,
            degree,
            focus,
        )

    @cute.jit
    def _stage_operands(
        self,
        grad_out: cute.Tensor,
        frame_contract: cute.Tensor,
        residual_scale: cute.Tensor,
        s_a: cute.Tensor,
        s_b: cute.Tensor,
        tidx: cutlass.Int32,
        m_base: cutlass.Int32,
        n_base: cutlass.Int32,
        degree: cutlass.Int32,
        focus: cutlass.Int32,
        k_base: cutlass.Constexpr[int],
    ):
        a_slots = (
            FRAME_CONTRACT_M_TILE * FRAME_CONTRACT_K_TILE + FRAME_CONTRACT_THREADS - 1
        ) // FRAME_CONTRACT_THREADS
        for slot in cutlass.range_constexpr(a_slots):
            linear = tidx + slot * FRAME_CONTRACT_THREADS
            row = linear // FRAME_CONTRACT_K_TILE
            k = linear - row * FRAME_CONTRACT_K_TILE
            node = m_base + row
            value = cutlass.Float32(0.0)
            if node < grad_out.shape[0]:
                output_channel = k_base + k
                value = grad_out[node, degree, focus, output_channel].to(
                    cutlass.Float32
                ) * residual_scale[focus, output_channel].to(cutlass.Float32)
            s_a[row, k] = value

        b_slots = (
            FRAME_CONTRACT_N_TILE * FRAME_CONTRACT_K_TILE + FRAME_CONTRACT_THREADS - 1
        ) // FRAME_CONTRACT_THREADS
        for slot in cutlass.range_constexpr(b_slots):
            linear = tidx + slot * FRAME_CONTRACT_THREADS
            output_column = linear // FRAME_CONTRACT_K_TILE
            k = linear - output_column * FRAME_CONTRACT_K_TILE
            logical_column = n_base + output_column
            value = cutlass.Float32(0.0)
            if logical_column < FRAME_CONTRACT_WIDTH:
                frame = logical_column // CHANNELS
                channel = logical_column - frame * CHANNELS
                value = frame_contract[
                    degree,
                    frame,
                    channel,
                    k_base + k,
                ].to(cutlass.Float32)
            s_b[output_column, k] = value
        cute.arch.sync_threads()

    @cute.jit
    def _store_compact_epilogue(
        self,
        coefficient_slab: cute.Tensor,
        accumulator: cute.Tensor,
        thr_mma,
        output_reference: cute.Tensor,
        m_base: cutlass.Int32,
        n_base: cutlass.Int32,
        degree: cutlass.Int32,
        focus: cutlass.Int32,
    ):
        accumulator.store(accumulator.load())
        identity = cute.make_identity_tensor(output_reference.shape)
        coordinates = thr_mma.partition_C(identity)
        for value_idx in range(cute.size(accumulator.shape)):
            coordinate = coordinates[value_idx]
            node = m_base + coordinate[0]
            logical_column = n_base + coordinate[1]
            if (
                node < coefficient_slab.shape[0]
                and logical_column < FRAME_CONTRACT_WIDTH
            ):
                frame = logical_column // CHANNELS
                channel = logical_column - frame * CHANNELS
                packed = degree * FRAME_COUNT + frame
                hidden = focus * CHANNELS + channel
                coefficient_slab[node, packed, hidden] = accumulator[value_idx].to(
                    cutlass.Float32
                )


class _FusedReadoutAdjoint:
    """Fuse gate, normalized-Gaunt adjoint, and FrameExpand transpose."""

    def __init__(self, schedule: Sm90GauntSchedule) -> None:
        self.schedule = schedule

    @cute.jit
    def __call__(
        self,
        coefficient_slab: cute.Tensor,
        scalar_gate: cute.Tensor,
        product: cute.Tensor,
        left: cute.Tensor,
        right: cute.Tensor,
        frame_expand_t: cute.Tensor,
        grad_query: cute.Tensor,
        grad_context: cute.Tensor,
        grad_scalar_gate: cute.Tensor,
        grad_scalar_out: cute.Tensor,
        stream: CUstream,
    ):
        workspace_layout = cute.make_layout(
            (WORKSPACE_PANELS, PACKED_COEFF_DIM, FOLDED_CHANNELS),
            stride=(VALUES_PER_PANEL, FOLDED_CHANNELS, 1),
        )
        statistic_layout = cute.make_layout(
            (READOUT_GROUPS, FOLDED_CHANNELS),
            stride=(FOLDED_CHANNELS, 1),
        )
        self.kernel(
            coefficient_slab,
            scalar_gate,
            product,
            left,
            right,
            frame_expand_t,
            grad_query,
            grad_context,
            grad_scalar_gate,
            grad_scalar_out,
            workspace_layout,
            statistic_layout,
        ).launch(
            grid=(left.shape[0], 1, 1),
            block=[READOUT_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        coefficient_slab: cute.Tensor,
        scalar_gate: cute.Tensor,
        product: cute.Tensor,
        left: cute.Tensor,
        right: cute.Tensor,
        frame_expand_t: cute.Tensor,
        grad_query: cute.Tensor,
        grad_context: cute.Tensor,
        grad_scalar_gate: cute.Tensor,
        grad_scalar_out: cute.Tensor,
        workspace_layout: cute.Layout,
        statistic_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        node, _, _ = cute.arch.block_idx()
        group = tidx >> 6
        folded_channel = tidx & (FOLDED_CHANNELS - 1)
        gate = scalar_gate[node, folded_channel].to(cutlass.Float32)

        smem = cute_utils.SmemAllocator()
        workspace = smem.allocate_tensor(cutlass.Float32, workspace_layout, 16)
        statistic_partials = smem.allocate_tensor(
            cutlass.Float32,
            statistic_layout,
            16,
        )
        statistic = cutlass.Float32(0.0)
        scalar_out = cutlass.Float32(0.0)

        for operand in cutlass.range_constexpr(3):
            for slot in cutlass.range_constexpr(ROWS_PER_GROUP):
                row = group * ROWS_PER_GROUP + slot
                value = left[node, row, folded_channel].to(cutlass.Float32)
                if cutlass.const_expr(operand == RIGHT_PANEL):
                    value = right[node, row, folded_channel].to(cutlass.Float32)
                if cutlass.const_expr(operand == GATED_COEFFICIENT_PANEL):
                    value = coefficient_slab[node, row, folded_channel].to(
                        cutlass.Float32
                    )
                    statistic = statistic + value * product[
                        node,
                        row,
                        folded_channel,
                    ].to(cutlass.Float32)
                    if row == 0:
                        scalar_out = value
                    value = value * gate
                workspace[operand, row, folded_channel] = value

        statistic_partials[group, folded_channel] = statistic
        cute.arch.sync_threads()

        if group == 0:
            total = statistic_partials[0, folded_channel]
            for statistic_group in cutlass.range_constexpr(
                1,
                READOUT_GROUPS,
                1,
            ):
                total = (
                    total
                    + statistic_partials[
                        statistic_group,
                        folded_channel,
                    ]
                )
            grad_scalar_gate[node, folded_channel] = total
            grad_scalar_out[node, folded_channel] = scalar_out

        if group == 0:
            self._accumulate_group(workspace, folded_channel, 0)
        elif group == 1:
            self._accumulate_group(workspace, folded_channel, 1)
        elif group == 2:
            self._accumulate_group(workspace, folded_channel, 2)
        else:
            self._accumulate_group(workspace, folded_channel, 3)
        cute.arch.sync_threads()

        for slot in cutlass.range_constexpr(DEGREES_PER_THREAD):
            linear = tidx + slot * READOUT_THREADS
            input_channel = linear & (CHANNELS - 1)
            degree = linear >> 5
            accumulator_left_0 = cutlass.Float32(0.0)
            accumulator_left_1 = cutlass.Float32(0.0)
            accumulator_right_0 = cutlass.Float32(0.0)
            accumulator_right_1 = cutlass.Float32(0.0)
            for frame in cutlass.range_constexpr(FRAME_COUNT):
                packed_row = degree * FRAME_COUNT + frame
                for output_channel in cutlass.range_constexpr(CHANNELS):
                    weight = frame_expand_t[
                        degree,
                        frame,
                        output_channel,
                        input_channel,
                    ].to(cutlass.Float32)
                    accumulator_left_0 = accumulator_left_0 + (
                        workspace[GRAD_LEFT_PANEL, packed_row, output_channel] * weight
                    )
                    accumulator_right_0 = accumulator_right_0 + (
                        workspace[GRAD_RIGHT_PANEL, packed_row, output_channel] * weight
                    )
                    accumulator_left_1 = accumulator_left_1 + (
                        workspace[
                            GRAD_LEFT_PANEL,
                            packed_row,
                            CHANNELS + output_channel,
                        ]
                        * weight
                    )
                    accumulator_right_1 = accumulator_right_1 + (
                        workspace[
                            GRAD_RIGHT_PANEL,
                            packed_row,
                            CHANNELS + output_channel,
                        ]
                        * weight
                    )
            grad_query[node, degree, 0, input_channel] = accumulator_left_0
            grad_query[node, degree, 1, input_channel] = accumulator_left_1
            grad_context[node, degree, 0, input_channel] = accumulator_right_0
            grad_context[node, degree, 1, input_channel] = accumulator_right_1

    @cute.jit
    def _accumulate_group(
        self,
        workspace: cute.Tensor,
        folded_channel: cutlass.Int32,
        group: cutlass.Constexpr,
    ):
        rows = self.schedule.output_groups[group]
        for row_slot in cutlass.range_constexpr(len(rows)):
            input_row = rows[row_slot]
            accumulator_left = cutlass.Float32(0.0)
            accumulator_right = cutlass.Float32(0.0)
            terms = self.schedule.rows[input_row]
            for path in cutlass.range_constexpr(len(terms)):
                (
                    left_row,
                    right_row,
                    _,
                    left_weight_value,
                    right_weight_value,
                ) = terms[path]
                common = (
                    cutlass.Float32(left_weight_value)
                    * workspace[GATED_COEFFICIENT_PANEL, left_row, folded_channel]
                )
                accumulator_left = accumulator_left + (
                    common * workspace[RIGHT_PANEL, right_row, folded_channel]
                )
                accumulator_right = accumulator_right + (
                    common * workspace[LEFT_PANEL, right_row, folded_channel]
                )
                if cutlass.const_expr(left_row != right_row):
                    mirrored_common = (
                        cutlass.Float32(right_weight_value)
                        * workspace[
                            GATED_COEFFICIENT_PANEL,
                            right_row,
                            folded_channel,
                        ]
                    )
                    accumulator_left = accumulator_left + (
                        mirrored_common
                        * workspace[RIGHT_PANEL, left_row, folded_channel]
                    )
                    accumulator_right = accumulator_right + (
                        mirrored_common
                        * workspace[LEFT_PANEL, left_row, folded_channel]
                    )
            workspace[GRAD_LEFT_PANEL, input_row, folded_channel] = accumulator_left
            workspace[GRAD_RIGHT_PANEL, input_row, folded_channel] = accumulator_right


def _fake_state() -> cute.Tensor:
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), DEGREE_COUNT, FOCUS_COUNT, CHANNELS),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_frame_weight() -> cute.Tensor:
    return make_fake_compact_tensor(
        cutlass.Float32,
        (DEGREE_COUNT, FRAME_COUNT, CHANNELS, CHANNELS),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_focus_channel() -> cute.Tensor:
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_packed() -> cute.Tensor:
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), PACKED_COEFF_DIM, FOLDED_CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_node_channel() -> cute.Tensor:
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), FOLDED_CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )


@device_aware_lru_cache(maxsize=4)
def _compiled_frame_contract_adjoint() -> Callable:
    return cute.compile(
        _FrameContractAdjoint(),
        _fake_state(),
        _fake_frame_weight(),
        _fake_focus_channel(),
        _fake_packed(),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@device_aware_lru_cache(maxsize=8)
def _compiled_fused_readout_adjoint(schedule: Sm90GauntSchedule) -> Callable:
    packed = _fake_packed()
    node_channel = _fake_node_channel()
    state = _fake_state()
    return cute.compile(
        _FusedReadoutAdjoint(schedule),
        packed,
        node_channel,
        packed,
        packed,
        packed,
        _fake_frame_weight(),
        state,
        state,
        node_channel,
        node_channel,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be FP32")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be at least 16-byte aligned")


def _validate_runtime(
    grad_out: torch.Tensor,
    scalar_gate: torch.Tensor,
    product: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    state: Sm90MessageGridState,
) -> int:
    device = left.device
    if (
        device.type != "cuda"
        or tuple(torch.cuda.get_device_capability(device)) != SM90_CAPABILITY
    ):
        raise RuntimeError("the fused message-grid readout requires SM90")
    if not uses_strict_fp32_matmul():
        raise RuntimeError("the fused message-grid readout requires strict FP32")
    nodes = int(left.shape[0])
    if nodes <= 0:
        raise ValueError("the fused message-grid readout requires N > 0")
    packed_shape = (nodes, PACKED_COEFF_DIM, FOLDED_CHANNELS)
    state_shape = (nodes, DEGREE_COUNT, FOCUS_COUNT, CHANNELS)
    node_channel_shape = (nodes, FOLDED_CHANNELS)
    for name, tensor, shape in (
        ("grad_out", grad_out, state_shape),
        ("scalar_gate", scalar_gate, node_channel_shape),
        ("product", product, packed_shape),
        ("left", left, packed_shape),
        ("right", right, packed_shape),
        (
            "frame_contract",
            state.frame_contract,
            (DEGREE_COUNT, FRAME_COUNT, CHANNELS, CHANNELS),
        ),
        ("residual_scale", state.residual_scale, (FOCUS_COUNT, CHANNELS)),
        (
            "frame_expand_t",
            state.frame_expand_t,
            (DEGREE_COUNT, FRAME_COUNT, CHANNELS, CHANNELS),
        ),
    ):
        _validate_tensor(name, tensor, shape, device)
    return nodes


def run_sm90_message_grid_backward(
    net: Any,
    query_flat: torch.Tensor,
    context_flat: torch.Tensor,
    grad_out_flat: torch.Tensor,
    product_flat: torch.Tensor,
    state: Sm90MessageGridState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query/context adjoints without expanded global adjoint slabs."""
    from ..message_grid import (
        _as_flat_ndfc,
        _focus_linear_backward_input,
        _frame_expand_packed,
        _swiglu_backward_input,
        _validate_contract,
    )

    query, context = _validate_contract(net, query_flat, context_flat)
    nodes = int(query_flat.shape[0])
    scalar_pair = torch.cat([query[:, 0], context[:, 0]], dim=-1).to(net.dtype)
    scalar_gate = (
        torch.sigmoid(net.scalar_gate(scalar_pair))
        .reshape(
            nodes,
            FOLDED_CHANNELS,
        )
        .contiguous()
    )
    left = _frame_expand_packed(net.frame_expand, query).view(
        nodes,
        PACKED_COEFF_DIM,
        FOLDED_CHANNELS,
    )
    right = _frame_expand_packed(net.frame_expand, context).view(
        nodes,
        PACKED_COEFF_DIM,
        FOLDED_CHANNELS,
    )
    grad_out = (
        _as_flat_ndfc(
            net,
            "grad_out",
            grad_out_flat,
            like=query_flat,
        )
        .to(net.dtype)
        .contiguous()
    )
    _validate_runtime(grad_out, scalar_gate, product_flat, left, right, state)

    coefficient_slab = torch.empty_like(left)
    grad_query = torch.empty(
        query.shape,
        device=query.device,
        dtype=query.dtype,
    )
    grad_context = torch.empty(
        context.shape,
        device=context.device,
        dtype=context.dtype,
    )
    grad_scalar_gate = torch.empty_like(scalar_gate)
    grad_scalar_out = torch.empty_like(scalar_gate)
    with torch.cuda.device(left.device):
        _compiled_frame_contract_adjoint()(
            grad_out,
            state.frame_contract,
            state.residual_scale,
            coefficient_slab,
        )
        _compiled_fused_readout_adjoint(state.schedule)(
            coefficient_slab,
            scalar_gate,
            product_flat,
            left,
            right,
            state.frame_expand_t,
            grad_query,
            grad_context,
            grad_scalar_gate,
            grad_scalar_out,
        )

    gate = scalar_gate.view(nodes, FOCUS_COUNT, CHANNELS)
    grad_scalar_logits = grad_scalar_gate.view_as(gate) * gate * (1.0 - gate)
    grad_scalar_pair = _focus_linear_backward_input(
        net.scalar_gate,
        grad_scalar_logits,
    ) + _swiglu_backward_input(
        scalar_pair,
        grad_scalar_out.view(nodes, FOCUS_COUNT, CHANNELS),
    )
    grad_query[:, 0].add_(grad_scalar_pair[:, :, :CHANNELS])
    grad_context[:, 0].add_(grad_scalar_pair[:, :, CHANNELS:])
    return (
        grad_query.reshape_as(query_flat).to(dtype=query_flat.dtype),
        grad_context.reshape_as(context_flat).to(dtype=context_flat.dtype),
    )


__all__ = [
    "Sm90MessageGridState",
    "prepare_sm90_message_grid_state",
    "run_sm90_message_grid_backward",
]
