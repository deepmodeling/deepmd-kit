# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""One-launch strict-FP32 Neo SO2 gate/residual forward."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cute_utils
import torch
from cutlass.cute.runtime import (
    make_fake_stream,
    make_fake_tensor,
)

from ...compile_cache import (
    device_aware_lru_cache,
)
from ...runtime_policy import (
    FUSED_SO2_GATE_CAPABILITIES,
)

# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN201, ANN202, ANN204


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


TILE_M = 64
TILE_K = 16
THREADS = 256
STAGES = 3
FOCUS_COUNT = 2
M0_WIDTH = 4 * 32
PAIR_WIDTH = 6 * 32
FULL_WIDTH = M0_WIDTH + PAIR_WIDTH
DEFAULT_STREAM = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT)


def _supports_combined_forward(compute_capability: tuple[int, int]) -> bool:
    return compute_capability in FUSED_SO2_GATE_CAPABILITIES


def _require_16_byte_alignment(tensors: tuple[torch.Tensor, ...]) -> None:
    if any(tensor.data_ptr() % 16 for tensor in tensors):
        raise ValueError("combined Neo SO2 gate tensors must be 16-byte aligned")


@cute.jit
def _sigmoid(value):
    return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-value))


def _fake_focus_tensor(width: int):
    return make_fake_tensor(
        cutlass.Float32,
        (cute.sym_int32(), FOCUS_COUNT, width),
        (FOCUS_COUNT * width, width, 1),
        assumed_align=16,
    )


def _fake_focus_weight(width: int):
    return make_fake_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, width, width),
        (width * width, width, 1),
        assumed_align=16,
    )


def prepare_neo_so2_gate_combined_weights(
    w0: torch.Tensor,
    wp: torch.Tensor,
    gate_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack immutable weights into contiguous focus-major tensors."""
    return (
        w0.transpose(1, 2).contiguous(),
        wp.transpose(1, 2).contiguous(),
        gate_weight.permute(1, 0, 2).contiguous(),
    )


class CuteNeoSO2GateCombined:
    """M64/SO26/T256/S3 SIMT SGEMMs with one CTA-resident gate epilogue."""

    def __init__(self) -> None:
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=THREADS,
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mResidual: cute.Tensor,
        mW0: cute.Tensor,
        mWP: cute.Tensor,
        mGate: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream = DEFAULT_STREAM,
    ):
        sA_layout = cute.make_layout(
            (TILE_M, TILE_K, STAGES),
            stride=(1, TILE_M + 4, TILE_K * (TILE_M + 4)),
        )
        sB_pair_layout = cute.make_layout(
            (PAIR_WIDTH, TILE_K, STAGES),
            stride=(1, PAIR_WIDTH + 4, TILE_K * (PAIR_WIDTH + 4)),
        )
        sB_m0_layout = cute.make_layout(
            (M0_WIDTH, TILE_K, STAGES),
            stride=(1, PAIR_WIDTH + 4, TILE_K * (PAIR_WIDTH + 4)),
        )
        sY0_layout = cute.make_layout(
            (TILE_M, 32),
            stride=(32, 1),
        )
        sGate_layout = cute.make_layout(
            (TILE_M, 3 * 32),
            stride=(3 * 32, 1),
        )

        copy_layout = cute.make_layout(
            (THREADS // TILE_K, TILE_K),
            stride=(TILE_K, 1),
        )
        copy_value = cute.make_layout((1, 1))
        copy_a = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=mX.element_type.width,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mW0.element_type,
            num_bits_per_copy=mW0.element_type.width,
        )
        tiled_copy_A = cute.make_tiled_copy_tv(copy_a, copy_layout, copy_value)
        tiled_copy_B = cute.make_tiled_copy_tv(copy_b, copy_layout, copy_value)

        atoms_layout = cute.make_layout(
            (THREADS // 16, 16, 1),
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
        m0_op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        pair_op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        tiled_mma_m0 = cute.make_tiled_mma(
            m0_op,
            atoms_layout,
            permutation_mnk=(permutation_m, permutation_n, None),
        )
        tiled_mma_pair = cute.make_tiled_mma(
            pair_op,
            atoms_layout,
            permutation_mnk=(permutation_m, permutation_n, None),
        )

        self.kernel(
            mX,
            mResidual,
            mW0,
            mWP,
            mGate,
            mY,
            mOut,
            sA_layout,
            sB_m0_layout,
            sB_pair_layout,
            sY0_layout,
            sGate_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma_m0,
            tiled_mma_pair,
        ).launch(
            grid=(cute.ceil_div(mY.shape[0], TILE_M), FOCUS_COUNT, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mResidual: cute.Tensor,
        mW0: cute.Tensor,
        mWP: cute.Tensor,
        mGate: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        sA_layout: cute.Layout,
        sB_m0_layout: cute.Layout,
        sB_pair_layout: cute.Layout,
        sY0_layout: cute.Layout,
        sGate_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma_m0: cute.TiledMma,
        tiled_mma_pair: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        edge_tile, focus, _ = cute.arch.block_idx()

        x_focus = mX[None, focus, None]
        residual_focus = mResidual[None, focus, None]
        y_focus = mY[None, focus, None]
        out_focus = mOut[None, focus, None]
        matrix_layout_m0 = cute.make_layout(
            (mY.shape[0], M0_WIDTH),
            stride=(FOCUS_COUNT * FULL_WIDTH, 1),
        )
        matrix_layout_pair = cute.make_layout(
            (mY.shape[0], PAIR_WIDTH),
            stride=(FOCUS_COUNT * FULL_WIDTH, 1),
        )
        mA0 = cute.make_tensor(x_focus.iterator, matrix_layout_m0)
        mAPair = cute.make_tensor(
            x_focus.iterator + M0_WIDTH,
            matrix_layout_pair,
        )
        mR0 = cute.make_tensor(residual_focus.iterator, matrix_layout_m0)
        mRPair = cute.make_tensor(
            residual_focus.iterator + M0_WIDTH,
            matrix_layout_pair,
        )
        mY0 = cute.make_tensor(y_focus.iterator, matrix_layout_m0)
        mYPair = cute.make_tensor(
            y_focus.iterator + M0_WIDTH,
            matrix_layout_pair,
        )
        mOut0 = cute.make_tensor(out_focus.iterator, matrix_layout_m0)
        mOutPair = cute.make_tensor(
            out_focus.iterator + M0_WIDTH,
            matrix_layout_pair,
        )
        w0_focus = mW0[focus, None, None]
        wp_focus = mWP[focus, None, None]
        gate_focus = mGate[focus, None, None]

        smem = cute_utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
        sB = smem.allocate_tensor(cutlass.Float32, sB_pair_layout, 16)
        sY0 = smem.allocate_tensor(cutlass.Float32, sY0_layout, 16)
        sGate = smem.allocate_tensor(cutlass.Float32, sGate_layout, 16)

        self._run_m0(
            mA0,
            w0_focus,
            mR0,
            gate_focus,
            mY0,
            mOut0,
            sA,
            sB,
            sY0,
            sGate,
            sB_m0_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma_m0,
            tidx,
            edge_tile,
        )
        cute.arch.sync_threads()
        self._run_pair(
            mAPair,
            wp_focus,
            mRPair,
            mYPair,
            mOutPair,
            sA,
            sB,
            sGate,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma_pair,
            tidx,
            edge_tile,
        )

    @cute.jit
    def _run_m0(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mGate: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sY0: cute.Tensor,
        sGate: cute.Tensor,
        sB_m0_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        edge_tile: cutlass.Int32,
    ):
        sB_m0 = cute.make_tensor(sB.iterator, sB_m0_layout)
        self._run_gemm(
            mA,
            mB,
            mR,
            mGate,
            mY,
            mOut,
            sA,
            sB_m0,
            sY0,
            sGate,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            tidx,
            edge_tile,
            m0_block=True,
        )

    @cute.jit
    def _run_pair(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sGate: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        edge_tile: cutlass.Int32,
    ):
        self._run_gemm(
            mA,
            mB,
            mR,
            mB,
            mY,
            mOut,
            sA,
            sB,
            sA,
            sGate,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            tidx,
            edge_tile,
            m0_block=False,
        )

    @cute.jit
    def _run_gemm(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mGate: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sY0: cute.Tensor,
        sGate: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        edge_tile: cutlass.Int32,
        m0_block: cutlass.Constexpr[bool],
    ):
        width = M0_WIDTH if cutlass.const_expr(m0_block) else PAIR_WIDTH
        cta_tiler = (TILE_M, width, TILE_K)
        tiler_coord = (edge_tile, 0, None)
        thr_mma = tiled_mma.get_slice(tidx)

        gA = cute.local_tile(
            mA,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gR = cute.local_tile(
            mR,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        gY = cute.local_tile(
            mY,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        gOut = cute.local_tile(
            mOut,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        mcA = cute.make_identity_tensor(mA.shape)
        cA = cute.local_tile(
            mcA,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        tAcA = thr_copy_A.partition_S(cA)
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for row in range(tApA.shape[1]):
                tApA[rest_v, row, 0] = cute.elem_less(
                    tAcA[(0, rest_v), row, 0, 0][0],
                    mA.shape[0],
                )

        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )
        for k_tile in range(1, k_pipe_max - 1):
            if k_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile],
                )
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < k_tile_count
                else cutlass.Int32(0)
            )
            cute.arch.cp_async_commit_group()

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgR = thr_mma.partition_C(gR)
        tCgY = thr_mma.partition_C(gY)
        tCgOut = thr_mma.partition_C(gOut)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgOut)
        tCrC.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)
        tiles_issued = cutlass.Int32(k_pipe_max - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])

        if k_block_max > 1:
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(k_pipe_max - 2)
                    self.cta_sync_barrier.arrive_and_wait()

                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_p[None, None, k_block_next],
                    tCrB[None, None, k_block_next],
                )
                if k_block == 0:
                    if tiles_issued < k_tile_count:
                        cute.copy(
                            tiled_copy_A,
                            tAgA[None, None, None, gmem_pipe_read],
                            tAsA[None, None, None, smem_pipe_write],
                            pred=tApA,
                        )
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )
                if k_block == 0:
                    if tiles_issued < k_tile_count:
                        cute.copy(
                            tiled_copy_B,
                            tBgB[None, None, None, gmem_pipe_read],
                            tBsB[None, None, None, smem_pipe_write],
                        )
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(1)
                    )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        tCrC.store(tCrC.load())

        cC = cute.make_identity_tensor(gOut.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
        residue_m = mOut.shape[0] - cutlass.Int32(TILE_M) * edge_tile
        for idx in range(cute.size(tCrC.shape)):
            predC[idx] = cute.elem_less(tCpC[idx], (residue_m, width))

        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mOut.element_type,
        )
        cute.copy(atom, tCrC, tCgY, pred=predC)
        tCrR = tiled_mma.make_fragment_C(tCgR)
        tCrR.fill(0.0)
        cute.copy(atom, tCgR, tCrR, pred=predC)

        if cutlass.const_expr(m0_block):
            for idx in range(cute.size(tCrC.shape)):
                if predC[idx]:
                    local_row = tCpC[idx][0]
                    local_col = tCpC[idx][1]
                    if local_col < 32:
                        sY0[local_row, local_col] = tCrC[idx].to(cutlass.Float32)
            cute.arch.sync_threads()

            gate_slots = (TILE_M * 3 * 32 + THREADS - 1) // THREADS
            for slot in cutlass.range_constexpr(gate_slots):
                linear_idx = tidx + slot * THREADS
                if linear_idx < TILE_M * 3 * 32:
                    local_row = linear_idx // (3 * 32)
                    gate_idx = linear_idx - local_row * (3 * 32)
                    global_row = edge_tile * TILE_M + local_row
                    if global_row < mOut.shape[0]:
                        gate_logit = cutlass.Float32(0.0)
                        for k in cutlass.range_constexpr(32):
                            gate_logit += sY0[local_row, k] * mGate[k, gate_idx]
                        sGate[local_row, gate_idx] = _sigmoid(gate_logit)
            cute.arch.sync_threads()

        for idx in range(cute.size(tCrC.shape)):
            if predC[idx]:
                local_row = tCpC[idx][0]
                local_col = tCpC[idx][1]
                value = tCrC[idx].to(cutlass.Float32)
                if cutlass.const_expr(m0_block):
                    if local_col < 32:
                        value = value * _sigmoid(value)
                    else:
                        value = value * sGate[local_row, local_col - 32]
                else:
                    gate_idx = local_col
                    if gate_idx >= 3 * 32:
                        gate_idx = gate_idx - 3 * 32
                    value = value * sGate[local_row, gate_idx]
                tCrC[idx] = value + tCrR[idx].to(cutlass.Float32)

        cute.copy(atom, tCrC, tCgOut, pred=predC)


@device_aware_lru_cache(maxsize=8)
def _compile_combined_forward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    if not _supports_combined_forward(compute_capability):
        raise RuntimeError("combined forward requires a supported compute capability")
    with torch.cuda.device(device_index):
        fake_x = _fake_focus_tensor(FULL_WIDTH)
        fake_residual = _fake_focus_tensor(FULL_WIDTH)
        fake_w0 = _fake_focus_weight(M0_WIDTH)
        fake_wp = _fake_focus_weight(PAIR_WIDTH)
        fake_gate = make_fake_tensor(
            cutlass.Float32,
            (FOCUS_COUNT, 32, 3 * 32),
            (32 * 3 * 32, 3 * 32, 1),
            assumed_align=16,
        )
        fake_y = _fake_focus_tensor(FULL_WIDTH)
        fake_out = _fake_focus_tensor(FULL_WIDTH)
        operation = CuteNeoSO2GateCombined()
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        return cute.compile(
            operation,
            fake_x,
            fake_residual,
            fake_w0,
            fake_wp,
            fake_gate,
            fake_y,
            fake_out,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )


class CuteNeoSO2GateCombinedFwdRunner:
    """Prebuilt one-launch forward that writes full y and no global aux."""

    def __init__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        y: torch.Tensor,
        out: torch.Tensor,
        *,
        packed_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        packed_weights_ready: tuple[torch.cuda.Event, int],
    ) -> None:
        expected = (x.shape[0], FOCUS_COUNT, 10, 32)
        if tuple(x.shape) != expected:
            raise ValueError(f"x must have shape {expected}, got {tuple(x.shape)}")
        if residual.shape != x.shape or y.shape != x.shape or out.shape != x.shape:
            raise ValueError("residual, y, and out must match x")
        if x.shape[0] <= 0:
            raise ValueError("combined Neo SO2 gate forward requires E > 0")
        tensors = (x, residual, y, out)
        if any(
            tensor.dtype != torch.float32 or not tensor.is_cuda for tensor in tensors
        ):
            raise TypeError(
                "combined Neo SO2 gate forward requires CUDA float32 tensors"
            )
        if any(tensor.device != x.device for tensor in tensors):
            raise ValueError(
                "all combined Neo SO2 gate forward tensors must share x.device"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError(
                "combined Neo SO2 gate forward requires canonical contiguous tensors"
            )
        _require_16_byte_alignment(tensors)
        device_index = x.device.index
        if device_index is None:
            raise RuntimeError("combined Neo SO2 gate forward requires a CUDA index")
        compute_capability = tuple(torch.cuda.get_device_capability(device_index))
        if not _supports_combined_forward(compute_capability):
            raise RuntimeError(
                "combined forward requires a supported compute capability"
            )

        packed_w0, packed_wp, packed_gate = packed_weights
        packed = (packed_w0, packed_wp, packed_gate)
        if any(
            tensor.dtype != torch.float32 or not tensor.is_cuda for tensor in packed
        ):
            raise TypeError(
                "packed combined forward weights must be CUDA float32 tensors"
            )
        if any(tensor.device != x.device for tensor in packed):
            raise ValueError("all packed combined forward weights must share x.device")
        if any(not tensor.is_contiguous() for tensor in packed):
            raise ValueError("packed combined forward weights must be contiguous")
        _require_16_byte_alignment(packed)
        if tuple(packed_w0.shape) != (FOCUS_COUNT, M0_WIDTH, M0_WIDTH):
            raise ValueError("packed w0 must have shape (2,128,128)")
        if tuple(packed_wp.shape) != (FOCUS_COUNT, PAIR_WIDTH, PAIR_WIDTH):
            raise ValueError("packed wp must have shape (2,192,192)")
        if tuple(packed_gate.shape) != (FOCUS_COUNT, 32, 3 * 32):
            raise ValueError("packed gate weight must have shape (2,32,96)")

        with torch.cuda.device(x.device):
            self._compiled = _compile_combined_forward(
                device_index,
                compute_capability,
            )
        self._args = (
            x.reshape(x.shape[0], FOCUS_COUNT, FULL_WIDTH),
            residual.reshape(x.shape[0], FOCUS_COUNT, FULL_WIDTH),
            packed_w0,
            packed_wp,
            packed_gate,
            y.reshape(x.shape[0], FOCUS_COUNT, FULL_WIDTH),
            out.reshape(x.shape[0], FOCUS_COUNT, FULL_WIDTH),
        )
        self._device = x.device
        self._packed_weights_ready = packed_weights_ready
        self._packed_weights_waited_streams: set[int] = set()
        self.y = y
        self.out = out

    def __call__(self) -> torch.Tensor:
        with torch.cuda.device(self._device):
            torch_stream = torch.cuda.current_stream(self._device)
            ready_event, producer_stream = self._packed_weights_ready
            if (
                torch_stream.cuda_stream != producer_stream
                and torch_stream.cuda_stream not in self._packed_weights_waited_streams
            ):
                torch_stream.wait_event(ready_event)
                self._packed_weights_waited_streams.add(torch_stream.cuda_stream)
            stream = cuda.CUstream(torch_stream.cuda_stream)
            self._compiled(*self._args, stream=stream)
        return self.out
