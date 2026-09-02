# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Direct destination statistics for the SM90 final SO2Linear/Phase-C boundary.

One CTA owns one ``(node, focus)`` segment.  Its feature-owning threads keep
all output-degree statistics in registers while walking the destination's
edges in 64-edge chunks.  The CTA writes the final node-scale ``a0`` and
``a1`` statistics directly, eliminating the global chunk partials and their
second reduction launch.

The strict-FP32 node GEMMs are applied only after edge values have been
reduced to node-scale sufficient statistics.
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
# ruff: noqa: ANN001, ANN202, ANN204, TC002

EDGE_CHUNK = 64
FOCUS_COUNT = 2
CHANNELS = 32
DEGREE_COUNT = 16
M0_WIDTH = 128
M1_WIDTH = 96
PACKED_WIGNER_VALUES = 46
M0_THREADS = M0_WIDTH
M1_THREADS = M1_WIDTH * 2
THREADS = M0_THREADS + M1_THREADS
M1_SCALE_VALUES = EDGE_CHUNK * (DEGREE_COUNT - 1) * 2
M0_SCALE_VALUES = EDGE_CHUNK * DEGREE_COUNT
TOTAL_SCALE_VALUES = M0_SCALE_VALUES + M1_SCALE_VALUES
LOADS_PER_THREAD = (TOTAL_SCALE_VALUES + THREADS - 1) // THREADS
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@dataclass(frozen=True)
class ExpandedFinalWeights:
    """Dense final SO2Linear blocks selected for each output degree."""

    w0: torch.Tensor
    wc: torch.Tensor


@dataclass(frozen=True)
class ExpandedComplexWorkspace:
    """Node-scale real and complex sufficient statistics."""

    m0: torch.Tensor
    m1: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in (self.m0, self.m1)
        )


def prepare_expanded_final_weights(
    w0: torch.Tensor,
    wc: torch.Tensor,
) -> ExpandedFinalWeights:
    """Select the dense input block needed by each full output degree."""
    if tuple(w0.shape) != (FOCUS_COUNT, M0_WIDTH, M0_WIDTH):
        raise ValueError("w0 must have shape (2,128,128)")
    if tuple(wc.shape) != (FOCUS_COUNT, M1_WIDTH, M1_WIDTH):
        raise ValueError("wc must have shape (2,96,96)")
    if w0.dtype != torch.float32 or wc.dtype != torch.complex64:
        raise TypeError("w0/wc must be float32/complex64")
    degree_by_q = (0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3)
    blocks0 = torch.stack(
        [
            w0[:, :, degree * CHANNELS : (degree + 1) * CHANNELS]
            for degree in degree_by_q
        ],
        dim=1,
    ).contiguous()
    blocks1 = torch.stack(
        [
            wc[:, :, (degree - 1) * CHANNELS : degree * CHANNELS]
            for degree in degree_by_q[1:]
        ],
        dim=1,
    ).contiguous()
    return ExpandedFinalWeights(w0=blocks0, wc=blocks1)


def _require_sm90_strict_fp32(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("SM90 final Phase C requires CUDA")
    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise RuntimeError("SM90 final Phase C requires compute capability 9.0")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict FP32 requires float32 matmul precision 'highest'")


@dataclass(frozen=True)
class DirectStatisticsForwardResult:
    """Forward output and the final node-scale sufficient statistics."""

    output: torch.Tensor
    workspace: ExpandedComplexWorkspace

    @property
    def statistics_storage_bytes(self) -> int:
        return self.workspace.storage_bytes


@cute.jit
def _packed_index_runtime(q, row_slot):
    base = cutlass.Int32(0)
    width = cutlass.Int32(1)
    local = cutlass.Int32(0)
    if q >= 9:
        base = cutlass.Int32(25)
        width = cutlass.Int32(7)
        local = q - 9
    elif q >= 4:
        base = cutlass.Int32(10)
        width = cutlass.Int32(5)
        local = q - 4
    elif q >= 1:
        base = cutlass.Int32(1)
        width = cutlass.Int32(3)
        local = q - 1
    return base + row_slot * width + local


class CuteDirectNodeStatistics:
    """Accumulate all real and complex statistics in one destination CTA."""

    @cute.jit
    def __call__(
        self,
        m0: cute.Tensor,
        m1_ri: cute.Tensor,
        dt_packed: cute.Tensor,
        beta: cute.Tensor,
        dst_ptr: cute.Tensor,
        a0: cute.Tensor,
        a1_ri: cute.Tensor,
        stream: CUstream,
    ):
        self.kernel(m0, m1_ri, dt_packed, beta, dst_ptr, a0, a1_ri).launch(
            grid=[a0.shape[2], FOCUS_COUNT, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        m0: cute.Tensor,
        m1_ri: cute.Tensor,
        dt_packed: cute.Tensor,
        beta: cute.Tensor,
        dst_ptr: cute.Tensor,
        a0: cute.Tensor,
        a1_ri: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        node, focus, _ = cute.arch.block_idx()
        node_lo = dst_ptr[node]
        node_hi = dst_ptr[node + 1]
        edge_count = node_hi - node_lo
        chunk_count = (edge_count + EDGE_CHUNK - 1) // EDGE_CHUNK

        smem = cutlass.utils.SmemAllocator()
        m0_scale_storage = smem.allocate_tensor(cutlass.Float32, M0_SCALE_VALUES)
        m0_scales = cute.make_tensor(
            m0_scale_storage.iterator,
            cute.make_layout(
                (EDGE_CHUNK, DEGREE_COUNT),
                stride=(DEGREE_COUNT, 1),
            ),
        )
        m1_scale_storage = smem.allocate_tensor(cutlass.Float32, M1_SCALE_VALUES)
        m1_scales = cute.make_tensor(
            m1_scale_storage.iterator,
            cute.make_layout(
                (EDGE_CHUNK, DEGREE_COUNT - 1, 2),
                stride=((DEGREE_COUNT - 1) * 2, 2, 1),
            ),
        )

        # A single 16-value register fragment serves either a real feature or
        # one component of a complex feature.  Complex threads use entries
        # [0, 15), avoiding two live accumulator arrays in generated code.
        accumulators = cute.make_rmem_tensor(
            cute.make_layout((DEGREE_COUNT,), stride=(1,)),
            cutlass.Float32,
        )
        accumulators.fill(0.0)

        for chunk_slot in cutlass.range(chunk_count, unroll=1):
            lo = node_lo + chunk_slot * EDGE_CHUNK
            hi = lo + EDGE_CHUNK
            if node_hi < hi:
                hi = node_hi

            for load_slot in cutlass.range_constexpr(LOADS_PER_THREAD):
                linear = tidx + load_slot * THREADS
                if linear < M0_SCALE_VALUES:
                    edge_slot = linear // DEGREE_COUNT
                    q = linear - edge_slot * DEGREE_COUNT
                    edge = lo + edge_slot
                    value = cutlass.Float32(0.0)
                    if edge < hi:
                        panel = _packed_index_runtime(q, cutlass.Int32(0))
                        value = beta[edge, focus].to(cutlass.Float32) * dt_packed[
                            edge, panel
                        ].to(cutlass.Float32)
                    m0_scales[edge_slot, q] = value
                elif linear < TOTAL_SCALE_VALUES:
                    item = linear - M0_SCALE_VALUES
                    edge_slot = item // ((DEGREE_COUNT - 1) * 2)
                    remainder = item - edge_slot * (DEGREE_COUNT - 1) * 2
                    q1 = remainder // 2
                    component = remainder - q1 * 2
                    edge = lo + edge_slot
                    value = cutlass.Float32(0.0)
                    if edge < hi:
                        panel = _packed_index_runtime(q1 + 1, component + 1)
                        value = beta[edge, focus].to(cutlass.Float32) * dt_packed[
                            edge, panel
                        ].to(cutlass.Float32)
                    m1_scales[edge_slot, q1, component] = value
            cute.arch.sync_threads()

            if tidx < M0_THREADS:
                feature = tidx
                for edge_slot in cutlass.range_constexpr(EDGE_CHUNK):
                    edge = lo + edge_slot
                    if edge < hi:
                        x = m0[focus, edge, feature].to(cutlass.Float32)
                        for q in cutlass.range_constexpr(DEGREE_COUNT):
                            value = accumulators[q].to(cutlass.Float32)
                            value += m0_scales[edge_slot, q] * x
                            accumulators[q] = value
            elif tidx < THREADS:
                complex_thread = tidx - M0_THREADS
                feature = complex_thread // 2
                component = complex_thread - feature * 2
                for edge_slot in cutlass.range_constexpr(EDGE_CHUNK):
                    edge = lo + edge_slot
                    if edge < hi:
                        xr = m1_ri[focus, edge, feature, 0].to(cutlass.Float32)
                        xi = m1_ri[focus, edge, feature, 1].to(cutlass.Float32)
                        for q1 in cutlass.range_constexpr(DEGREE_COUNT - 1):
                            dr = m1_scales[edge_slot, q1, 0]
                            di = m1_scales[edge_slot, q1, 1]
                            value = accumulators[q1].to(cutlass.Float32)
                            if component == 0:
                                value += dr * xr + di * xi
                            else:
                                value += dr * xi - di * xr
                            accumulators[q1] = value

            # Every thread must finish reading shared scales before the next
            # 64-edge chunk overwrites them.
            cute.arch.sync_threads()

        if tidx < M0_THREADS:
            feature = tidx
            for q in cutlass.range_constexpr(DEGREE_COUNT):
                a0[focus, q, node, feature] = accumulators[q]
        elif tidx < THREADS:
            complex_thread = tidx - M0_THREADS
            feature = complex_thread // 2
            component = complex_thread - feature * 2
            for q1 in cutlass.range_constexpr(DEGREE_COUNT - 1):
                a1_ri[focus, q1, node, feature, component] = accumulators[q1]


def _fake_m0_edges():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, cute.sym_int64(), M0_WIDTH),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_m1_edges_ri():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, cute.sym_int64(), M1_WIDTH, 2),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_dt():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), PACKED_WIGNER_VALUES),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_beta():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), FOCUS_COUNT),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_index():
    return make_fake_compact_tensor(
        cutlass.Int32,
        (cute.sym_int64(),),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )


def _fake_a0():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, DEGREE_COUNT, cute.sym_int64(), M0_WIDTH),
        stride_order=(3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )


def _fake_a1_ri():
    return make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, DEGREE_COUNT - 1, cute.sym_int64(), M1_WIDTH, 2),
        stride_order=(4, 3, 2, 1, 0),
        **FAKE_TENSOR_KW,
    )


@device_aware_lru_cache(maxsize=2)
def _compiled_direct_statistics() -> Callable:
    return cute.compile(
        CuteDirectNodeStatistics(),
        _fake_m0_edges(),
        _fake_m1_edges_ri(),
        _fake_dt(),
        _fake_beta(),
        _fake_index(),
        _fake_a0(),
        _fake_a1_ri(),
        stream=make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _validate_forward_inputs(
    m0: torch.Tensor,
    m1: torch.Tensor,
    dt_packed: torch.Tensor,
    beta: torch.Tensor,
    dst_ptr: torch.Tensor,
) -> tuple[torch.device, int]:
    device = m0.device
    _require_sm90_strict_fp32(device)
    edge_count = int(m0.shape[1])
    node_count = int(dst_ptr.numel() - 1)
    expected = (
        ("m0", m0, (FOCUS_COUNT, edge_count, M0_WIDTH), torch.float32),
        ("m1", m1, (FOCUS_COUNT, edge_count, M1_WIDTH), torch.complex64),
        (
            "dt_packed",
            dt_packed,
            (edge_count, PACKED_WIGNER_VALUES),
            torch.float32,
        ),
        ("beta", beta, (edge_count, FOCUS_COUNT), torch.float32),
        ("dst_ptr", dst_ptr, (node_count + 1,), torch.int32),
    )
    for name, tensor, shape, dtype in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {device}")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be at least 16-byte aligned")
    return device, node_count


def run_direct_statistics_forward(
    *,
    m0: torch.Tensor,
    m1: torch.Tensor,
    dt_packed: torch.Tensor,
    beta: torch.Tensor,
    dst_ptr: torch.Tensor,
    weights: ExpandedFinalWeights,
    edge_chunk: int = EDGE_CHUNK,
    chunk_slots: int | None = None,
) -> DirectStatisticsForwardResult:
    """Build final node statistics directly and apply node-scale SO2Linear.

    ``edge_chunk`` and ``chunk_slots`` retain the chunked-forward argument
    contract used by the SM90 K1 runner.
    Only the 64-edge internal schedule is supported; ``chunk_slots`` is not an
    allocation dimension in this implementation.
    """
    if edge_chunk != EDGE_CHUNK:
        raise ValueError(f"direct statistics requires edge_chunk={EDGE_CHUNK}")
    if chunk_slots is not None and chunk_slots <= 0:
        raise ValueError("chunk_slots must be positive when provided")
    device, node_count = _validate_forward_inputs(m0, m1, dt_packed, beta, dst_ptr)
    a0 = torch.empty(
        (FOCUS_COUNT, DEGREE_COUNT, node_count, M0_WIDTH),
        device=device,
        dtype=torch.float32,
    )
    a1 = torch.empty(
        (FOCUS_COUNT, DEGREE_COUNT - 1, node_count, M1_WIDTH),
        device=device,
        dtype=torch.complex64,
    )
    with torch.cuda.device(device):
        _compiled_direct_statistics()(
            m0,
            torch.view_as_real(m1),
            dt_packed,
            beta,
            dst_ptr,
            a0,
            torch.view_as_real(a1),
        )
        out0 = torch.bmm(a0.flatten(0, 1), weights.w0.flatten(0, 1)).view(
            FOCUS_COUNT, DEGREE_COUNT, node_count, CHANNELS
        )
        out1 = torch.bmm(a1.flatten(0, 1), weights.wc.flatten(0, 1)).view(
            FOCUS_COUNT, DEGREE_COUNT - 1, node_count, CHANNELS
        )
        output = out0.permute(2, 0, 1, 3).contiguous()
        output[:, :, 1:] += out1.real.permute(2, 0, 1, 3)
    return DirectStatisticsForwardResult(
        output=output,
        workspace=ExpandedComplexWorkspace(m0=a0, m1=a1),
    )


__all__ = [
    "EDGE_CHUNK",
    "THREADS",
    "DirectStatisticsForwardResult",
    "ExpandedFinalWeights",
    "prepare_expanded_final_weights",
    "run_direct_statistics_forward",
]
