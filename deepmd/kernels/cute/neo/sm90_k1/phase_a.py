# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Direct strict-FP32 Phase-A producer for the persistent-complex SO2 stack.

The generic boundary materializes Neo's reduced SO2 state as block-real
``(E,2,10,32)`` and then launches a second kernel to transpose/split it into
focus-major ``m=0`` real and interleaved ``m=1`` complex panels. This producer
applies the same packed-Wigner rotation, compact radial
maps, and rank-1 channel basis, but writes the persistent representation
directly:

* reduced rows 0..3 -> ``m0[focus, edge, 4 * channel]``;
* reduced rows 4..6 -> the real component of ``m1``;
* reduced rows 7..9 -> the imaginary component of ``m1``.

No full-edge block-real slab exists on this path. ``N`` and ``E``
remain runtime dimensions; only the Neo representation contract is static.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import torch
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from ..compile_cache import (
    device_aware_lru_cache,
)
from ..k1_wigner_layout import (
    PACKED_VALUE_COUNT,
)
from .persistent import (
    NeoPersistentComplexState,
    validate_neo_persistent_complex_state,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN202, ANN204

EDGE_TILE = 32
THREADS = 256
FOCUS_COUNT = 2
FOCUS_DIM = 32
FULL_CHANNELS = FOCUS_COUNT * FOCUS_DIM
M0_ROWS = 4
M1_ROWS = 3
M0_WIDTH = M0_ROWS * FOCUS_DIM
M1_WIDTH = M1_ROWS * FOCUS_DIM
RADIAL_COMPACT = 25

D_CACHE_BYTES = EDGE_TILE * PACKED_VALUE_COUNT * 4
RADIAL_CACHE_BYTES = EDGE_TILE * RADIAL_COMPACT * 4
SRC_CACHE_BYTES = EDGE_TILE * 4
BASIS_CACHE_BYTES = FULL_CHANNELS * 4
CTA_SHARED_BYTES = (
    D_CACHE_BYTES + RADIAL_CACHE_BYTES + SRC_CACHE_BYTES + BASIS_CACHE_BYTES
)

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}
DEFAULT_STREAM = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT)

__all__ = [
    "CTA_SHARED_BYTES",
    "run_neo_phase_a_persistent_complex_fp32",
]


def _fake_x_wide():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), 16 * FULL_CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_src():
    return make_fake_compact_tensor(
        cutlass.Int32,
        (cute.sym_int64(),),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )


def _fake_edge_matrix(columns: int):
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), columns),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_channel_basis():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FULL_CHANNELS,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )


def _fake_m0():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, cute.sym_int64(), M0_WIDTH),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_m1_ri():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, cute.sym_int64(), M1_WIDTH, 2),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )


@cute.jit
def _rotation_row_cached(
    x_wide,
    d_cache,
    edge_row,
    src_node,
    channel,
    panel_start: cutlass.Constexpr[int],
    full_start: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
):
    """Evaluate one packed-Wigner row with the Phase-A reduction order."""
    value = cutlass.Float32(0.0)
    for local_col in cutlass.range_constexpr(width):
        value += d_cache[edge_row, panel_start + local_col].to(
            cutlass.Float32
        ) * x_wide[
            src_node,
            (full_start + local_col) * FULL_CHANNELS + channel,
        ].to(cutlass.Float32)
    return value


class CuteNeoPhaseAPersistentComplexFP32:
    """Produce native persistent-complex panels without a dense boundary."""

    @cute.jit
    def __call__(
        self,
        x_wide,
        src,
        d_full,
        radial_compact,
        channel_basis,
        m0,
        m1_ri,
        stream: cuda.CUstream = DEFAULT_STREAM,
    ):
        d_layout = cute.make_layout(
            (EDGE_TILE, PACKED_VALUE_COUNT),
            stride=(PACKED_VALUE_COUNT, 1),
        )
        radial_layout = cute.make_layout(
            (EDGE_TILE, RADIAL_COMPACT),
            stride=(RADIAL_COMPACT, 1),
        )
        edge_layout = cute.make_layout((EDGE_TILE,), stride=(1,))
        basis_layout = cute.make_layout((FULL_CHANNELS,), stride=(1,))
        self.kernel(
            x_wide,
            src,
            d_full,
            radial_compact,
            channel_basis,
            m0,
            m1_ri,
            d_layout,
            radial_layout,
            edge_layout,
            basis_layout,
        ).launch(
            grid=(cute.ceil_div(m0.shape[1], EDGE_TILE), 1, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_wide,
        src,
        d_full,
        radial_compact,
        channel_basis,
        m0,
        m1_ri,
        d_layout,
        radial_layout,
        edge_layout,
        basis_layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        edge_tile, _, _ = cute.arch.block_idx()
        edge_count = m0.shape[1]

        smem = cute_utils.SmemAllocator()
        d_cache = smem.allocate_tensor(cutlass.Float32, d_layout, 16)
        radial_cache = smem.allocate_tensor(cutlass.Float32, radial_layout, 16)
        src_cache = smem.allocate_tensor(cutlass.Int32, edge_layout, 16)
        basis_cache = smem.allocate_tensor(cutlass.Float32, basis_layout, 16)

        self._load_edge_state(
            src,
            d_full,
            radial_compact,
            channel_basis,
            d_cache,
            radial_cache,
            src_cache,
            basis_cache,
            edge_count,
            tidx,
            edge_tile,
        )
        self._produce_split_panels(
            x_wide,
            d_cache,
            radial_cache,
            src_cache,
            basis_cache,
            m0,
            m1_ri,
            edge_count,
            tidx,
            edge_tile,
        )

    @cute.jit
    def _load_edge_state(
        self,
        src,
        d_full,
        radial_compact,
        channel_basis,
        d_cache,
        radial_cache,
        src_cache,
        basis_cache,
        edge_count,
        tidx,
        edge_tile,
    ):
        d_slots = (EDGE_TILE * PACKED_VALUE_COUNT + THREADS - 1) // THREADS
        for slot in cutlass.range_constexpr(d_slots):
            linear = tidx + slot * THREADS
            if linear < EDGE_TILE * PACKED_VALUE_COUNT:
                edge_row = linear // PACKED_VALUE_COUNT
                column = linear - edge_row * PACKED_VALUE_COUNT
                edge = edge_tile * EDGE_TILE + edge_row
                value = cutlass.Float32(0.0)
                if edge < edge_count:
                    value = d_full[edge, column].to(cutlass.Float32)
                d_cache[edge_row, column] = value

        radial_slots = (EDGE_TILE * RADIAL_COMPACT + THREADS - 1) // THREADS
        for slot in cutlass.range_constexpr(radial_slots):
            linear = tidx + slot * THREADS
            if linear < EDGE_TILE * RADIAL_COMPACT:
                edge_row = linear // RADIAL_COMPACT
                column = linear - edge_row * RADIAL_COMPACT
                edge = edge_tile * EDGE_TILE + edge_row
                value = cutlass.Float32(0.0)
                if edge < edge_count:
                    value = radial_compact[edge, column].to(cutlass.Float32)
                radial_cache[edge_row, column] = value

        if tidx < EDGE_TILE:
            edge = edge_tile * EDGE_TILE + tidx
            src_node = cutlass.Int32(0)
            if edge < edge_count:
                src_node = src[edge]
            src_cache[tidx] = src_node
        if tidx < FULL_CHANNELS:
            basis_cache[tidx] = channel_basis[tidx].to(cutlass.Float32)
        cute.arch.sync_threads()

    @cute.jit
    def _produce_split_panels(
        self,
        x_wide,
        d_cache,
        radial_cache,
        src_cache,
        basis_cache,
        m0,
        m1_ri,
        edge_count,
        tidx,
        edge_tile,
    ):
        tasks = (EDGE_TILE * FULL_CHANNELS) // THREADS
        for task in cutlass.range_constexpr(tasks):
            linear = tidx + task * THREADS
            edge_row = linear // FULL_CHANNELS
            channel = linear - edge_row * FULL_CHANNELS
            edge = edge_tile * EDGE_TILE + edge_row

            if edge < edge_count:
                focus = channel // FOCUS_DIM
                focus_channel = channel - focus * FOCUS_DIM
                src_node = src_cache[edge_row]
                x0 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 0, 0, 1
                )
                x1 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 1, 1, 3
                )
                x2 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 10, 4, 5
                )
                x3 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 25, 9, 7
                )
                x4 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 4, 1, 3
                )
                x5 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 15, 4, 5
                )
                x6 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 32, 9, 7
                )
                x7 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 7, 1, 3
                )
                x8 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 20, 4, 5
                )
                x9 = _rotation_row_cached(
                    x_wide, d_cache, edge_row, src_node, channel, 39, 9, 7
                )

                basis = basis_cache[channel].to(cutlass.Float32)
                y0 = (
                    radial_cache[edge_row, 0] * x0
                    + radial_cache[edge_row, 4] * x1
                    + radial_cache[edge_row, 8] * x2
                    + radial_cache[edge_row, 12] * x3
                ) * basis
                y1 = (
                    radial_cache[edge_row, 1] * x0
                    + radial_cache[edge_row, 5] * x1
                    + radial_cache[edge_row, 9] * x2
                    + radial_cache[edge_row, 13] * x3
                ) * basis
                y2 = (
                    radial_cache[edge_row, 2] * x0
                    + radial_cache[edge_row, 6] * x1
                    + radial_cache[edge_row, 10] * x2
                    + radial_cache[edge_row, 14] * x3
                ) * basis
                y3 = (
                    radial_cache[edge_row, 3] * x0
                    + radial_cache[edge_row, 7] * x1
                    + radial_cache[edge_row, 11] * x2
                    + radial_cache[edge_row, 15] * x3
                ) * basis
                y4 = (
                    radial_cache[edge_row, 16] * x4
                    + radial_cache[edge_row, 19] * x5
                    + radial_cache[edge_row, 22] * x6
                ) * basis
                y5 = (
                    radial_cache[edge_row, 17] * x4
                    + radial_cache[edge_row, 20] * x5
                    + radial_cache[edge_row, 23] * x6
                ) * basis
                y6 = (
                    radial_cache[edge_row, 18] * x4
                    + radial_cache[edge_row, 21] * x5
                    + radial_cache[edge_row, 24] * x6
                ) * basis
                y7 = (
                    radial_cache[edge_row, 16] * x7
                    + radial_cache[edge_row, 19] * x8
                    + radial_cache[edge_row, 22] * x9
                ) * basis
                y8 = (
                    radial_cache[edge_row, 17] * x7
                    + radial_cache[edge_row, 20] * x8
                    + radial_cache[edge_row, 23] * x9
                ) * basis
                y9 = (
                    radial_cache[edge_row, 18] * x7
                    + radial_cache[edge_row, 21] * x8
                    + radial_cache[edge_row, 24] * x9
                ) * basis

                m0[focus, edge, focus_channel] = y0
                m0[focus, edge, FOCUS_DIM + focus_channel] = y1
                m0[focus, edge, 2 * FOCUS_DIM + focus_channel] = y2
                m0[focus, edge, 3 * FOCUS_DIM + focus_channel] = y3
                m1_ri[focus, edge, focus_channel, 0] = y4
                m1_ri[focus, edge, focus_channel, 1] = y7
                m1_ri[focus, edge, FOCUS_DIM + focus_channel, 0] = y5
                m1_ri[focus, edge, FOCUS_DIM + focus_channel, 1] = y8
                m1_ri[focus, edge, 2 * FOCUS_DIM + focus_channel, 0] = y6
                m1_ri[focus, edge, 2 * FOCUS_DIM + focus_channel, 1] = y9


@device_aware_lru_cache(maxsize=8)
def _compiled_producer(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    if compute_capability != (9, 0):
        raise RuntimeError("direct split Phase A requires SM90")
    with torch.cuda.device(device_index):
        return cute.compile(
            CuteNeoPhaseAPersistentComplexFP32(),
            _fake_x_wide(),
            _fake_src(),
            _fake_edge_matrix(PACKED_VALUE_COUNT),
            _fake_edge_matrix(RADIAL_COMPACT),
            _fake_channel_basis(),
            _fake_m0(),
            _fake_m1_ri(),
            stream=make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )


def _validate_inputs(
    x_wide: torch.Tensor,
    src: torch.Tensor,
    d_full: torch.Tensor,
    radial_compact: torch.Tensor,
    channel_basis: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    if x_wide.ndim != 3 or tuple(x_wide.shape[1:]) != (16, FULL_CHANNELS):
        raise ValueError(f"x_wide must have shape (N,16,64), got {x_wide.shape}")
    if x_wide.shape[0] <= 0:
        raise ValueError("direct split Phase A requires N > 0")
    if src.ndim != 1 or src.dtype not in (torch.int32, torch.int64):
        raise TypeError("src must be a one-dimensional int32 or int64 tensor")
    edge_count = src.numel()
    if edge_count <= 0:
        raise ValueError("direct split Phase A requires E > 0")
    if tuple(d_full.shape) != (edge_count, PACKED_VALUE_COUNT):
        raise ValueError(f"d_full must have shape {(edge_count, PACKED_VALUE_COUNT)}")
    if tuple(radial_compact.shape) != (edge_count, RADIAL_COMPACT):
        raise ValueError(
            f"radial_compact must have shape {(edge_count, RADIAL_COMPACT)}"
        )
    if tuple(channel_basis.shape) != (FULL_CHANNELS,):
        raise ValueError("channel_basis must have shape (64,)")

    float_tensors = (x_wide, d_full, radial_compact, channel_basis)
    if any(t.dtype != torch.float32 or not t.is_cuda for t in float_tensors):
        raise TypeError("all Phase-A floating-point operands must be CUDA float32")
    if not src.is_cuda:
        raise TypeError("src must be a CUDA tensor")
    if any(t.device != x_wide.device for t in (*float_tensors[1:], src)):
        raise ValueError("all Phase-A operands must share x_wide.device")
    if any(not t.is_contiguous() for t in float_tensors):
        raise ValueError("all Phase-A floating-point operands must be contiguous")
    src_i32 = (
        src
        if src.dtype == torch.int32 and src.is_contiguous()
        else src.to(dtype=torch.int32).contiguous()
    )
    return edge_count, src_i32


def _allocate_state(edge_count: int, device: torch.device) -> NeoPersistentComplexState:
    return NeoPersistentComplexState(
        m0=torch.empty(
            (FOCUS_COUNT, edge_count, M0_WIDTH),
            dtype=torch.float32,
            device=device,
        ),
        m1=torch.empty(
            (FOCUS_COUNT, edge_count, M1_WIDTH),
            dtype=torch.complex64,
            device=device,
        ),
    )


def run_neo_phase_a_persistent_complex_fp32(
    *,
    x_wide: torch.Tensor,
    src: torch.Tensor,
    d_full: torch.Tensor,
    radial_compact: torch.Tensor,
    channel_basis: torch.Tensor,
    out: NeoPersistentComplexState | None = None,
) -> NeoPersistentComplexState:
    """Write Phase A directly into the persistent stack's native split state."""
    edge_count, src_i32 = _validate_inputs(
        x_wide,
        src,
        d_full,
        radial_compact,
        channel_basis,
    )
    if out is None:
        out = _allocate_state(edge_count, x_wide.device)
    validate_neo_persistent_complex_state(out, name="out")
    if out.edge_count != edge_count or out.m0.device != x_wide.device:
        raise ValueError("out must have matching E and share x_wide.device")

    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict FP32 requires float32 matmul precision 'highest'")
    device_index = x_wide.device.index
    if device_index is None:
        raise RuntimeError("direct split Phase A requires CUDA")
    compute_capability = tuple(torch.cuda.get_device_capability(device_index))
    compiled = _compiled_producer(device_index, compute_capability)
    m1_ri = torch.view_as_real(out.m1)
    if not m1_ri.is_contiguous():
        raise ValueError("out.m1 must expose a contiguous interleaved real/imag view")
    stream = cuda.CUstream(torch.cuda.current_stream(x_wide.device).cuda_stream)
    compiled(
        x_wide.view(x_wide.shape[0], 16 * FULL_CHANNELS),
        src_i32,
        d_full,
        radial_compact,
        channel_basis,
        out.m0,
        m1_ri,
        stream=stream,
    )
    return out
