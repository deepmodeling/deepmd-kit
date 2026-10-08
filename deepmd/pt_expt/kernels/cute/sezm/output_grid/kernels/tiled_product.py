# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tiled strict-FP32 Neo output-grid product forward and first backward."""

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
import cutlass.pipeline as pipeline
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from ... import (
    runtime_policy,
)

# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, TC002, TC003


PACKED_COEFF_DIM = 48
GRID_SIZE = 152
SUPPORTED_HIDDEN_CHANNELS = (96, 192)
TILE_M = 64
TILE_N = 64
TILE_K = 8
C96_TAIL_TILE_N = 32
C96_TAIL_CHANNEL_TILE = 2
SM80_C96_TILE_N = 48
SM80_C96_THREADS = 128
MMA_ATOMS_N = 16
THREADS = 128
STAGES = 3
GRID_TILES = 3
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


def _validate_hidden_channels(hidden_channels: int) -> int:
    hidden_channels = int(hidden_channels)
    if hidden_channels not in SUPPORTED_HIDDEN_CHANNELS:
        raise ValueError(
            "tiled output-grid channel width must be one of "
            f"{SUPPORTED_HIDDEN_CHANNELS}, got {hidden_channels}"
        )
    return hidden_channels


class TiledOutputGridProductForward:
    """Dual tiled projections, shared product, and tiled backprojection."""

    def __init__(
        self,
        hidden_channels: int = 192,
        *,
        tile_n: int = TILE_N,
        channel_tile_start: int = 0,
        channel_tile_count: int | None = None,
    ) -> None:
        self.hidden_channels = _validate_hidden_channels(hidden_channels)
        tile_n = int(tile_n)
        if tile_n not in (C96_TAIL_TILE_N, SM80_C96_TILE_N, TILE_N):
            raise ValueError("output-grid forward tile_n must be 32, 48, or 64")
        if tile_n == SM80_C96_TILE_N and self.hidden_channels != 96:
            raise ValueError("output-grid forward N=48 specializes C=96")
        channel_tile_start = int(channel_tile_start)
        total_channel_tiles = (self.hidden_channels + tile_n - 1) // tile_n
        if channel_tile_count is None:
            channel_tile_count = total_channel_tiles - channel_tile_start
        channel_tile_count = int(channel_tile_count)
        if (
            channel_tile_start < 0
            or channel_tile_count <= 0
            or channel_tile_start + channel_tile_count > total_channel_tiles
        ):
            raise ValueError("invalid output-grid forward channel-tile range")
        if tile_n == C96_TAIL_TILE_N and (
            self.hidden_channels != 96
            or channel_tile_start != C96_TAIL_CHANNEL_TILE
            or channel_tile_count != 1
        ):
            raise ValueError("output-grid forward N=32 specializes the C96 tail panel")
        self.cta_tiler = (TILE_M, tile_n, TILE_K)
        self.channel_tile_start = channel_tile_start
        self.channel_tiles = channel_tile_count
        self.has_channel_residue = self.hidden_channels % tile_n != 0
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=THREADS,
        )

    @cute.jit
    def __call__(
        self,
        left: cute.Tensor,
        right: cute.Tensor,
        to_grid: cute.Tensor,
        from_grid: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ):
        tile_n = self.cta_tiler[1]
        sA_layout = cute.make_layout(
            (TILE_M, TILE_K, STAGES),
            stride=(1, TILE_M + 4, TILE_K * (TILE_M + 4)),
        )
        sB_layout = cute.make_layout(
            (tile_n, TILE_K, STAGES),
            stride=(1, tile_n, TILE_K * tile_n),
        )
        product_layout = cute.make_layout(
            (GRID_SIZE, tile_n),
            stride=(tile_n, 1),
        )
        copy_a_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            left.element_type,
            num_bits_per_copy=left.element_type.width,
        )
        copy_a_layout = cute.make_layout(
            (THREADS // TILE_K, TILE_K),
            stride=(TILE_K, 1),
        )
        tiled_copy_A = cute.make_tiled_copy_tv(
            copy_a_atom,
            copy_a_layout,
            cute.make_layout((1, 1)),
        )
        vector = 2 if cutlass.const_expr(tile_n == C96_TAIL_TILE_N) else 4
        if cutlass.const_expr(tile_n == SM80_C96_TILE_N):
            copy_b_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                left.element_type,
                num_bits_per_copy=left.element_type.width,
            )
            copy_b_major = THREADS // TILE_K
            copy_b_layout = cute.make_layout(
                (copy_b_major, TILE_K),
                stride=(1, copy_b_major),
            )
            copy_b_value_layout = cute.make_layout(
                (tile_n // copy_b_major, 1),
            )
        else:
            copy_b_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                left.element_type,
                num_bits_per_copy=left.element_type.width * vector,
            )
            copy_b_major = tile_n // vector
            copy_b_layout = cute.make_layout(
                (copy_b_major, THREADS // copy_b_major),
                stride=(1, copy_b_major),
            )
            copy_b_value_layout = cute.make_layout((vector, 1))
        tiled_copy_B = cute.make_tiled_copy_tv(
            copy_b_atom,
            copy_b_layout,
            copy_b_value_layout,
        )
        atoms_layout = cute.make_layout(
            (THREADS // 16, 16, 1),
            stride=(16, 1, 0),
        )
        permutation_m = cute.make_layout(
            (atoms_layout.shape[0], 4),
            stride=(4, 1),
        )
        values_n = tile_n // MMA_ATOMS_N
        permutation_n = cute.make_layout(
            (atoms_layout.shape[1], values_n),
            stride=(values_n, 1),
        )
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.MmaUniversalOp(cutlass.Float32),
            atoms_layout,
            permutation_mnk=(permutation_m, permutation_n, None),
        )
        self.kernel(
            left,
            right,
            to_grid,
            from_grid,
            out,
            sA_layout,
            sB_layout,
            product_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=(left.shape[0], self.channel_tiles, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        left: cute.Tensor,
        right: cute.Tensor,
        to_grid: cute.Tensor,
        from_grid: cute.Tensor,
        out: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        product_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        node, channel_tile, _ = cute.arch.block_idx()
        channel_tile = channel_tile + self.channel_tile_start

        left_node = left[node, None, None]
        right_node = right[node, None, None]
        out_node = out[node, None, None]
        matrix_b_layout = cute.make_layout(
            (self.hidden_channels, PACKED_COEFF_DIM),
            stride=(1, self.hidden_channels),
        )
        left_b = cute.make_tensor(left_node.iterator, matrix_b_layout)
        right_b = cute.make_tensor(right_node.iterator, matrix_b_layout)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
        sB_left = smem.allocate_tensor(cutlass.Float32, sB_layout, 16)
        sB_right = smem.allocate_tensor(cutlass.Float32, sB_layout, 16)
        product = smem.allocate_tensor(cutlass.Float32, product_layout, 16)

        for grid_tile in cutlass.range_constexpr(GRID_TILES):
            self._dual_projection_product(
                to_grid,
                left_b,
                right_b,
                product,
                sA,
                sB_left,
                sB_right,
                tiled_copy_A,
                tiled_copy_B,
                tiled_mma,
                tidx,
                grid_tile,
                channel_tile,
            )

        product_b = cute.make_tensor(
            product.iterator,
            cute.make_layout(
                (self.cta_tiler[1], GRID_SIZE),
                stride=(1, self.cta_tiler[1]),
            ),
        )
        self._backproject(
            from_grid,
            product_b,
            out_node,
            sA,
            tiled_copy_A,
            tiled_mma,
            tidx,
            channel_tile,
        )

    @cute.jit
    def _dual_projection_product(
        self,
        mA: cute.Tensor,
        mB_left: cute.Tensor,
        mB_right: cute.Tensor,
        mProduct: cute.Tensor,
        sA: cute.Tensor,
        sB_left: cute.Tensor,
        sB_right: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        grid_tile: cutlass.Constexpr,
        channel_tile: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        gA = cute.local_tile(
            mA,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, None, 1),
        )
        gB_left = cute.local_tile(
            mB_left,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(None, 1, 1),
        )
        gB_right = cute.local_tile(
            mB_right,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(None, 1, 1),
        )
        gProduct = cute.local_tile(
            mProduct,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, 1, None),
        )

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB_left = thr_copy_B.partition_S(gB_left)
        tBgB_right = thr_copy_B.partition_S(gB_right)
        tBsB_left = thr_copy_B.partition_D(sB_left)
        tBsB_right = thr_copy_B.partition_D(sB_right)

        if cutlass.const_expr(self.has_channel_residue):
            cB = cute.local_tile(
                cute.make_identity_tensor(mB_left.shape),
                tiler=self.cta_tiler,
                coord=(0, channel_tile, None),
                proj=(None, 1, 1),
            )
            tBcB = thr_copy_B.partition_S(cB)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB_left.shape[0][1],
                        cute.size(tBsB_left, mode=[1]),
                        cute.size(tBsB_left, mode=[2]),
                    ),
                    stride=(cute.size(tBsB_left, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tBpB.shape[0]):
                for channel in range(tBpB.shape[1]):
                    tBpB[rest_v, channel, 0] = cute.elem_less(
                        tBcB[(0, rest_v), channel, 0, 0][0],
                        mB_left.shape[0],
                    )

        cA = cute.local_tile(
            cute.make_identity_tensor(mA.shape),
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
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
        if cutlass.const_expr(self.has_channel_residue):
            cute.copy(
                tiled_copy_B,
                tBgB_left[None, None, None, gmem_pipe_read],
                tBsB_left[None, None, None, 0],
                pred=tBpB,
            )
        else:
            cute.copy(
                tiled_copy_B,
                tBgB_left[None, None, None, gmem_pipe_read],
                tBsB_left[None, None, None, 0],
            )
        if cutlass.const_expr(self.has_channel_residue):
            cute.copy(
                tiled_copy_B,
                tBgB_right[None, None, None, gmem_pipe_read],
                tBsB_right[None, None, None, 0],
                pred=tBpB,
            )
        else:
            cute.copy(
                tiled_copy_B,
                tBgB_right[None, None, None, gmem_pipe_read],
                tBsB_right[None, None, None, 0],
            )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1
        for stage in range(1, STAGES - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            if cutlass.const_expr(self.has_channel_residue):
                cute.copy(
                    tiled_copy_B,
                    tBgB_left[None, None, None, gmem_pipe_read],
                    tBsB_left[None, None, None, stage],
                    pred=tBpB,
                )
            else:
                cute.copy(
                    tiled_copy_B,
                    tBgB_left[None, None, None, gmem_pipe_read],
                    tBsB_left[None, None, None, stage],
                )
            if cutlass.const_expr(self.has_channel_residue):
                cute.copy(
                    tiled_copy_B,
                    tBgB_right[None, None, None, gmem_pipe_read],
                    tBsB_right[None, None, None, stage],
                    pred=tBpB,
                )
            else:
                cute.copy(
                    tiled_copy_B,
                    tBgB_right[None, None, None, gmem_pipe_read],
                    tBsB_right[None, None, None, stage],
                )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA = thr_mma.partition_A(sA)
        tCsB_left = thr_mma.partition_B(sB_left)
        tCsB_right = thr_mma.partition_B(sB_right)
        tCgProduct = thr_mma.partition_C(gProduct)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB_left = tiled_mma.make_fragment_B(tCsB_left[None, None, None, 0])
        tCrB_right = tiled_mma.make_fragment_B(tCsB_right[None, None, None, 0])
        tCrLeft = tiled_mma.make_fragment_C(tCgProduct)
        tCrRight = tiled_mma.make_fragment_C(tCgProduct)
        tCrLeft.fill(0.0)
        tCrRight.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_left_p = tCsB_left[None, None, None, smem_pipe_read]
        tCsB_right_p = tCsB_right[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(
                tCsB_left_p[None, None, 0],
                tCrB_left[None, None, 0],
            )
            cute.autovec_copy(
                tCsB_right_p[None, None, 0],
                tCrB_right[None, None, 0],
            )

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_left_p = tCsB_left[None, None, None, smem_pipe_read]
                    tCsB_right_p = tCsB_right[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
                    self.cta_sync_barrier.arrive_and_wait()

                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_left_p[None, None, k_block_next],
                    tCrB_left[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_right_p[None, None, k_block_next],
                    tCrB_right[None, None, k_block_next],
                )
                if k_block == 0 and tiles_issued < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )

                cute.gemm(
                    tiled_mma,
                    tCrLeft,
                    tCrA[None, None, k_block],
                    tCrB_left[None, None, k_block],
                    tCrLeft,
                )
                cute.gemm(
                    tiled_mma,
                    tCrRight,
                    tCrA[None, None, k_block],
                    tCrB_right[None, None, k_block],
                    tCrRight,
                )

                if k_block == 0:
                    if tiles_issued < k_tile_count:
                        if cutlass.const_expr(self.has_channel_residue):
                            cute.copy(
                                tiled_copy_B,
                                tBgB_left[None, None, None, gmem_pipe_read],
                                tBsB_left[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        else:
                            cute.copy(
                                tiled_copy_B,
                                tBgB_left[None, None, None, gmem_pipe_read],
                                tBsB_left[None, None, None, smem_pipe_write],
                            )
                        if cutlass.const_expr(self.has_channel_residue):
                            cute.copy(
                                tiled_copy_B,
                                tBgB_right[None, None, None, gmem_pipe_read],
                                tBsB_right[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        else:
                            cute.copy(
                                tiled_copy_B,
                                tBgB_right[None, None, None, gmem_pipe_read],
                                tBsB_right[None, None, None, smem_pipe_write],
                            )
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == STAGES:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(0)
                    )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        cProduct = cute.make_identity_tensor(gProduct.shape)
        tCpProduct = thr_mma.partition_C(cProduct)
        pred = cute.make_rmem_tensor(tCrLeft.layout, cutlass.Boolean)
        residue_m = GRID_SIZE - TILE_M * grid_tile
        for idx in range(cute.size(tCrLeft.shape)):
            pred[idx] = cute.elem_less(
                tCpProduct[idx],
                (residue_m, self.cta_tiler[1]),
            )
            if pred[idx]:
                tCrLeft[idx] = tCrLeft[idx].to(cutlass.Float32) * tCrRight[idx].to(
                    cutlass.Float32
                )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mProduct.element_type,
        )
        cute.copy(atom, tCrLeft, tCgProduct, pred=pred)
        cute.arch.sync_threads()

    @cute.jit
    def _backproject(
        self,
        mA: cute.Tensor,
        mB_shared: cute.Tensor,
        mOut: cute.Tensor,
        sA: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        channel_tile: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        gA = cute.local_tile(
            mA,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(1, None, 1),
        )
        sB = cute.local_tile(
            mB_shared,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(None, 1, 1),
        )
        gOut = cute.local_tile(
            mOut,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(1, 1, None),
        )
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)

        cA = cute.local_tile(
            cute.make_identity_tensor(mA.shape),
            tiler=self.cta_tiler,
            coord=(0, 0, None),
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
                    PACKED_COEFF_DIM,
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
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1
        for stage in range(1, STAGES - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA = thr_mma.partition_A(sA)
        tSsB = thr_mma.partition_B(sB)
        tCgOut = thr_mma.partition_C(gOut)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tSsB[None, None, None, 0])
        tCrOut = tiled_mma.make_fragment_C(tCgOut)
        tCrOut.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        logical_k_tile = cutlass.Int32(0)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(
                tSsB[None, None, 0, logical_k_tile],
                tCrB[None, None, 0],
            )

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
                    self.cta_sync_barrier.arrive_and_wait()
                k_block_next = (k_block + 1) % k_block_max
                fragment_k_tile = logical_k_tile
                if k_block_max > 1:
                    if k_block == k_block_max - 1:
                        fragment_k_tile = (
                            logical_k_tile + 1
                            if logical_k_tile + 1 < k_tile_count
                            else logical_k_tile
                        )
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tSsB[None, None, k_block_next, fragment_k_tile],
                    tCrB[None, None, k_block_next],
                )
                if k_block == 0 and tiles_issued < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                cute.gemm(
                    tiled_mma,
                    tCrOut,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrOut,
                )
                if k_block == 0:
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == STAGES:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(0)
                    )
            logical_k_tile = logical_k_tile + 1

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        pred = cute.make_rmem_tensor(tCrOut.layout, cutlass.Boolean)
        if cutlass.const_expr(self.has_channel_residue):
            cOut = cute.local_tile(
                cute.make_identity_tensor(mOut.shape),
                tiler=self.cta_tiler,
                coord=(0, channel_tile, None),
                proj=(1, 1, None),
            )
            tCpOut = thr_mma.partition_C(cOut)
            for idx in range(cute.size(tCrOut.shape)):
                pred[idx] = cute.elem_less(tCpOut[idx], mOut.shape)
        else:
            cOut = cute.make_identity_tensor(gOut.shape)
            tCpOut = thr_mma.partition_C(cOut)
            for idx in range(cute.size(tCrOut.shape)):
                pred[idx] = cute.elem_less(
                    tCpOut[idx],
                    (PACKED_COEFF_DIM, self.cta_tiler[1]),
                )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mOut.element_type,
        )
        cute.copy(atom, tCrOut, tCgOut, pred=pred)


class TiledOutputGridProductBackward:
    """Tiled dP, projection recomputation, and dual coefficient adjoints."""

    def __init__(
        self,
        hidden_channels: int = 192,
        *,
        tile_n: int = TILE_N,
        channel_tile_start: int = 0,
        channel_tile_count: int | None = None,
    ) -> None:
        self.hidden_channels = _validate_hidden_channels(hidden_channels)
        self.tile_k = TILE_K
        tile_n = int(tile_n)
        if tile_n not in (C96_TAIL_TILE_N, SM80_C96_TILE_N, TILE_N):
            raise ValueError("output-grid backward tile_n must be 32, 48, or 64")
        if tile_n == SM80_C96_TILE_N and self.hidden_channels != 96:
            raise ValueError("output-grid N=48 specializes C=96")
        channel_tile_start = int(channel_tile_start)
        total_channel_tiles = (self.hidden_channels + tile_n - 1) // tile_n
        if channel_tile_count is None:
            channel_tile_count = total_channel_tiles - channel_tile_start
        channel_tile_count = int(channel_tile_count)
        if (
            channel_tile_start < 0
            or channel_tile_count <= 0
            or channel_tile_start + channel_tile_count > total_channel_tiles
        ):
            raise ValueError("invalid output-grid backward channel-tile range")
        if tile_n == C96_TAIL_TILE_N and (
            self.hidden_channels != 96
            or channel_tile_start != C96_TAIL_CHANNEL_TILE
            or channel_tile_count != 1
        ):
            raise ValueError(
                "output-grid backward N=32 specializes the C96 K=8 tail panel"
            )
        threads = SM80_C96_THREADS if tile_n == SM80_C96_TILE_N else THREADS
        self.sm80_c96_n48_panel = tile_n == SM80_C96_TILE_N
        self.cta_tiler = (TILE_M, tile_n, self.tile_k)
        self.channel_tile_start = channel_tile_start
        self.channel_tiles = channel_tile_count
        self.has_channel_residue = self.hidden_channels % tile_n != 0
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=threads,
        )

    @cute.jit
    def __call__(
        self,
        grad_out: cute.Tensor,
        left: cute.Tensor,
        right: cute.Tensor,
        to_grid: cute.Tensor,
        from_grid: cute.Tensor,
        grad_left: cute.Tensor,
        grad_right: cute.Tensor,
        stream: CUstream,
    ):
        tile_n = self.cta_tiler[1]
        threads = SM80_C96_THREADS if tile_n == SM80_C96_TILE_N else THREADS
        sA_row_layout = cute.make_layout(
            (TILE_M, self.tile_k, STAGES),
            stride=(1, TILE_M + 4, self.tile_k * (TILE_M + 4)),
        )
        sA_col_layout = cute.make_layout(
            (TILE_M, self.tile_k, STAGES),
            stride=(1, TILE_M, self.tile_k * TILE_M),
        )
        sB_layout = cute.make_layout(
            (tile_n, self.tile_k, STAGES),
            stride=(1, tile_n, self.tile_k * tile_n),
        )
        grid_layout = cute.make_layout(
            (GRID_SIZE, tile_n),
            stride=(tile_n, 1),
        )
        panel_layout = cute.make_layout(
            (TILE_M, tile_n),
            stride=(tile_n, 1),
        )

        copy_a_row_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            left.element_type,
            num_bits_per_copy=left.element_type.width,
        )
        copy_a_row_layout = cute.make_layout(
            (threads // self.tile_k, self.tile_k),
            stride=(self.tile_k, 1),
        )
        tiled_copy_A_row = cute.make_tiled_copy_tv(
            copy_a_row_atom,
            copy_a_row_layout,
            cute.make_layout((1, 1)),
        )

        vector = 4
        copy_a_col_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            left.element_type,
            num_bits_per_copy=left.element_type.width * vector,
        )
        copy_a_col_major = TILE_M // vector
        copy_a_col_layout = cute.make_layout(
            (copy_a_col_major, threads // copy_a_col_major),
            stride=(1, copy_a_col_major),
        )
        tiled_copy_A_col = cute.make_tiled_copy_tv(
            copy_a_col_atom,
            copy_a_col_layout,
            cute.make_layout((vector, 1)),
        )

        if cutlass.const_expr(tile_n == SM80_C96_TILE_N):
            # Keep all 128 threads in the copy/MMA contract. Each thread
            # issues three scalar N copies, exactly covering 48x8 without a
            # partial thread layout or an out-of-bounds 64-column staging tile.
            copy_b_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                left.element_type,
                num_bits_per_copy=left.element_type.width,
            )
            copy_b_major = threads // self.tile_k
            copy_b_layout = cute.make_layout(
                (copy_b_major, self.tile_k),
                stride=(1, copy_b_major),
            )
            copy_b_value_layout = cute.make_layout(
                (tile_n // copy_b_major, 1),
            )
        else:
            vector_b = 2 if cutlass.const_expr(tile_n == C96_TAIL_TILE_N) else vector
            copy_b_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                left.element_type,
                num_bits_per_copy=left.element_type.width * vector_b,
            )
            copy_b_major = tile_n // vector_b
            copy_b_layout = cute.make_layout(
                (copy_b_major, threads // copy_b_major),
                stride=(1, copy_b_major),
            )
            copy_b_value_layout = cute.make_layout((vector_b, 1))
        tiled_copy_B = cute.make_tiled_copy_tv(
            copy_b_atom,
            copy_b_layout,
            copy_b_value_layout,
        )

        # Follow the Ampere SGEMM thread topology: the universal-FMA atom is
        # always tiled over 16 threads in N. N=48 assigns three consecutive N
        # values to each thread. With 128 threads this produces a 32x48 MMA
        # tile, which divides the 64x48 CTA exactly; a 96-thread 24x48 tile
        # would create an unpredicated shared-memory fragment for rows 64..71.
        atoms_n = MMA_ATOMS_N
        atoms_m = threads // atoms_n
        values_n = tile_n // atoms_n
        atoms_layout = cute.make_layout(
            (atoms_m, atoms_n, 1),
            stride=(atoms_n, 1, 0),
        )
        permutation_m = cute.make_layout(
            (atoms_layout.shape[0], 4),
            stride=(4, 1),
        )
        permutation_n = cute.make_layout(
            (atoms_layout.shape[1], values_n),
            stride=(values_n, 1),
        )
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.MmaUniversalOp(cutlass.Float32),
            atoms_layout,
            permutation_mnk=(permutation_m, permutation_n, None),
        )

        self.kernel(
            grad_out,
            left,
            right,
            to_grid,
            from_grid,
            grad_left,
            grad_right,
            sA_row_layout,
            sA_col_layout,
            sB_layout,
            grid_layout,
            panel_layout,
            tiled_copy_A_row,
            tiled_copy_A_col,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=(left.shape[0], self.channel_tiles, 1),
            block=[threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        grad_out: cute.Tensor,
        left: cute.Tensor,
        right: cute.Tensor,
        to_grid: cute.Tensor,
        from_grid: cute.Tensor,
        grad_left: cute.Tensor,
        grad_right: cute.Tensor,
        sA_row_layout: cute.Layout,
        sA_col_layout: cute.Layout,
        sB_layout: cute.Layout,
        grid_layout: cute.Layout,
        panel_layout: cute.Layout,
        tiled_copy_A_row: cute.TiledCopy,
        tiled_copy_A_col: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        node, channel_tile, _ = cute.arch.block_idx()
        channel_tile = channel_tile + self.channel_tile_start

        matrix_b_layout = cute.make_layout(
            (self.hidden_channels, PACKED_COEFF_DIM),
            stride=(1, self.hidden_channels),
        )
        grad_out_b = cute.make_tensor(
            grad_out[node, None, None].iterator,
            matrix_b_layout,
        )
        left_b = cute.make_tensor(
            left[node, None, None].iterator,
            matrix_b_layout,
        )
        right_b = cute.make_tensor(
            right[node, None, None].iterator,
            matrix_b_layout,
        )
        grad_left_node = grad_left[node, None, None]
        grad_right_node = grad_right[node, None, None]
        from_grid_t = cute.make_tensor(
            from_grid.iterator,
            cute.make_layout(
                (GRID_SIZE, PACKED_COEFF_DIM),
                stride=(1, GRID_SIZE),
            ),
        )
        to_grid_t = cute.make_tensor(
            to_grid.iterator,
            cute.make_layout(
                (PACKED_COEFF_DIM, GRID_SIZE),
                stride=(1, PACKED_COEFF_DIM),
            ),
        )

        smem = cutlass.utils.SmemAllocator()
        sA_storage = smem.allocate_tensor(
            cutlass.Float32,
            sA_row_layout,
            16,
        )
        sA_row = sA_storage
        sA_col = cute.make_tensor(sA_storage.iterator, sA_col_layout)
        if cutlass.const_expr(self.sm80_c96_n48_panel):
            sB = smem.allocate_tensor(cutlass.Float32, sB_layout, 16)
            adjoint_panel = smem.allocate_tensor(
                cutlass.Float32,
                panel_layout,
                16,
            )
            self._panel_adjoint_backward(
                grad_out_b,
                left_b,
                right_b,
                to_grid,
                from_grid_t,
                to_grid_t,
                grad_left_node,
                grad_right_node,
                adjoint_panel,
                sA_row,
                sA_col,
                sB,
                tiled_copy_A_row,
                tiled_copy_A_col,
                tiled_copy_B,
                tiled_mma,
                tidx,
                channel_tile,
            )
        else:
            sB_left = smem.allocate_tensor(cutlass.Float32, sB_layout, 16)
            sB_right = smem.allocate_tensor(cutlass.Float32, sB_layout, 16)
            grad_left_grid = smem.allocate_tensor(cutlass.Float32, grid_layout, 16)
            grad_right_grid = smem.allocate_tensor(cutlass.Float32, grid_layout, 16)

            for grid_tile in cutlass.range_constexpr(GRID_TILES):
                self._single_projection_to_shared(
                    from_grid_t,
                    grad_out_b,
                    grad_left_grid,
                    sA_col,
                    sB_left,
                    tiled_copy_A_col,
                    tiled_copy_B,
                    tiled_mma,
                    tidx,
                    grid_tile,
                    channel_tile,
                )

            for grid_tile in cutlass.range_constexpr(GRID_TILES):
                self._dual_projection_adjoint(
                    to_grid,
                    left_b,
                    right_b,
                    grad_left_grid,
                    grad_right_grid,
                    sA_row,
                    sB_left,
                    sB_right,
                    tiled_copy_A_row,
                    tiled_copy_B,
                    tiled_mma,
                    tidx,
                    grid_tile,
                    channel_tile,
                )

            grad_left_grid_b = cute.make_tensor(
                grad_left_grid.iterator,
                cute.make_layout(
                    (self.cta_tiler[1], GRID_SIZE),
                    stride=(1, self.cta_tiler[1]),
                ),
            )
            grad_right_grid_b = cute.make_tensor(
                grad_right_grid.iterator,
                cute.make_layout(
                    (self.cta_tiler[1], GRID_SIZE),
                    stride=(1, self.cta_tiler[1]),
                ),
            )
            self._dual_backproject(
                to_grid_t,
                grad_left_grid_b,
                grad_right_grid_b,
                grad_left_node,
                grad_right_node,
                sA_col,
                tiled_copy_A_col,
                tiled_mma,
                tidx,
                channel_tile,
            )

    @cute.jit
    def _panel_adjoint_backward(
        self,
        grad_out_b: cute.Tensor,
        left_b: cute.Tensor,
        right_b: cute.Tensor,
        to_grid: cute.Tensor,
        from_grid_t: cute.Tensor,
        to_grid_t: cute.Tensor,
        grad_left_node: cute.Tensor,
        grad_right_node: cute.Tensor,
        adjoint_panel: cute.Tensor,
        sA_row: cute.Tensor,
        sA_col: cute.Tensor,
        sB: cute.Tensor,
        tiled_copy_A_row: cute.TiledCopy,
        tiled_copy_A_col: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        channel_tile: cutlass.Int32,
    ):
        """Keep dP in registers and reuse one shared adjoint panel."""
        thr_mma = tiled_mma.get_slice(tidx)
        gOut_left = cute.local_tile(
            grad_left_node,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(1, 1, None),
        )
        gOut_right = cute.local_tile(
            grad_right_node,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(1, 1, None),
        )
        tCgOut_left = thr_mma.partition_C(gOut_left)
        tCgOut_right = thr_mma.partition_C(gOut_right)
        tCrOut_left = tiled_mma.make_fragment_C(tCgOut_left)
        tCrOut_right = tiled_mma.make_fragment_C(tCgOut_right)
        tCrOut_left.fill(0.0)
        tCrOut_right.fill(0.0)

        panel_b = cute.make_tensor(
            adjoint_panel.iterator,
            cute.make_layout(
                (self.cta_tiler[1], TILE_M),
                stride=(1, self.cta_tiler[1]),
            ),
        )
        for grid_tile in cutlass.range_constexpr(GRID_TILES):
            tCrDP = self._projection_fragment(
                from_grid_t,
                grad_out_b,
                adjoint_panel,
                sA_col,
                sB,
                tiled_copy_A_col,
                tiled_copy_B,
                tiled_mma,
                tidx,
                grid_tile,
                channel_tile,
            )

            tCrRight_grid = self._projection_fragment(
                to_grid,
                right_b,
                adjoint_panel,
                sA_row,
                sB,
                tiled_copy_A_row,
                tiled_copy_B,
                tiled_mma,
                tidx,
                grid_tile,
                channel_tile,
            )
            self._store_adjoint_panel(
                tCrDP,
                tCrRight_grid,
                adjoint_panel,
                tiled_mma,
                tidx,
                grid_tile,
                channel_tile,
            )
            self._backproject_panel_accumulate(
                to_grid_t,
                panel_b,
                sA_col,
                tiled_copy_A_col,
                tiled_mma,
                tidx,
                grid_tile,
                tCrOut_left,
            )

            tCrLeft_grid = self._projection_fragment(
                to_grid,
                left_b,
                adjoint_panel,
                sA_row,
                sB,
                tiled_copy_A_row,
                tiled_copy_B,
                tiled_mma,
                tidx,
                grid_tile,
                channel_tile,
            )
            self._store_adjoint_panel(
                tCrDP,
                tCrLeft_grid,
                adjoint_panel,
                tiled_mma,
                tidx,
                grid_tile,
                channel_tile,
            )
            self._backproject_panel_accumulate(
                to_grid_t,
                panel_b,
                sA_col,
                tiled_copy_A_col,
                tiled_mma,
                tidx,
                grid_tile,
                tCrOut_right,
            )

        pred = cute.make_rmem_tensor(tCrOut_left.layout, cutlass.Boolean)
        if cutlass.const_expr(self.has_channel_residue):
            cOut = cute.local_tile(
                cute.make_identity_tensor(grad_left_node.shape),
                tiler=self.cta_tiler,
                coord=(0, channel_tile, None),
                proj=(1, 1, None),
            )
            tCpOut = thr_mma.partition_C(cOut)
            for idx in range(cute.size(tCrOut_left.shape)):
                pred[idx] = cute.elem_less(tCpOut[idx], grad_left_node.shape)
        else:
            cOut = cute.make_identity_tensor(gOut_left.shape)
            tCpOut = thr_mma.partition_C(cOut)
            for idx in range(cute.size(tCrOut_left.shape)):
                pred[idx] = cute.elem_less(
                    tCpOut[idx],
                    (PACKED_COEFF_DIM, self.cta_tiler[1]),
                )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            grad_left_node.element_type,
        )
        cute.copy(atom, tCrOut_left, tCgOut_left, pred=pred)
        cute.copy(atom, tCrOut_right, tCgOut_right, pred=pred)

    @cute.jit
    def _projection_fragment(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC_layout: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        grid_tile: cutlass.Constexpr,
        channel_tile: cutlass.Int32,
    ):
        """Project one 64-row grid panel and return its register fragment."""
        thr_mma = tiled_mma.get_slice(tidx)
        gA = cute.local_tile(
            mA,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(None, 1, 1),
        )
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        if cutlass.const_expr(self.has_channel_residue):
            cB = cute.local_tile(
                cute.make_identity_tensor(mB.shape),
                tiler=self.cta_tiler,
                coord=(0, channel_tile, None),
                proj=(None, 1, 1),
            )
            tBcB = thr_copy_B.partition_S(cB)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB.shape[0][1],
                        cute.size(tBsB, mode=[1]),
                        cute.size(tBsB, mode=[2]),
                    ),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tBpB.shape[0]):
                for channel in range(tBpB.shape[1]):
                    tBpB[rest_v, channel, 0] = cute.elem_less(
                        tBcB[(0, rest_v), channel, 0, 0][0],
                        mB.shape[0],
                    )

        cA = cute.local_tile(
            cute.make_identity_tensor(mA.shape),
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
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

        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA,
        )
        if cutlass.const_expr(self.has_channel_residue):
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, 0],
                pred=tBpB,
            )
        else:
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, 0],
            )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1
        for stage in range(1, STAGES - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            if cutlass.const_expr(self.has_channel_residue):
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, stage],
                    pred=tBpB,
                )
            else:
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, stage],
                )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(mC_layout)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
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
                if k_block == 0 and tiles_issued < k_tile_count:
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
                        if cutlass.const_expr(self.has_channel_residue):
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, gmem_pipe_read],
                                tBsB[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        else:
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, gmem_pipe_read],
                                tBsB[None, None, None, smem_pipe_write],
                            )
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == STAGES:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(0)
                    )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        return tCrC

    @cute.jit
    def _store_adjoint_panel(
        self,
        tCrDP: cute.Tensor,
        tCrBranch: cute.Tensor,
        panel: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        grid_tile: cutlass.Constexpr,
        channel_tile: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        tCgPanel = thr_mma.partition_C(panel)
        cPanel = cute.make_identity_tensor(panel.shape)
        tCpPanel = thr_mma.partition_C(cPanel)
        pred = cute.make_rmem_tensor(tCrBranch.layout, cutlass.Boolean)
        residue_m = GRID_SIZE - TILE_M * grid_tile
        residue_n = self.hidden_channels - self.cta_tiler[1] * channel_tile
        for idx in range(cute.size(tCrBranch.shape)):
            pred[idx] = cute.elem_less(
                tCpPanel[idx],
                (residue_m, residue_n),
            )
            if pred[idx]:
                tCrBranch[idx] = tCrDP[idx].to(cutlass.Float32) * tCrBranch[idx].to(
                    cutlass.Float32
                )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            panel.element_type,
        )
        cute.copy(atom, tCrBranch, tCgPanel, pred=pred)
        cute.arch.sync_threads()

    @cute.jit
    def _backproject_panel_accumulate(
        self,
        mA: cute.Tensor,
        panel_b: cute.Tensor,
        sA: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        grid_tile: cutlass.Constexpr,
        tCrOut: cute.Tensor,
    ):
        """Accumulate one 64-row adjoint panel into coefficient registers."""
        thr_mma = tiled_mma.get_slice(tidx)
        gA = cute.local_tile(
            mA,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(1, None, 1),
        )
        tAgA = tiled_copy_A.get_slice(tidx).partition_S(gA)
        tAsA = tiled_copy_A.get_slice(tidx).partition_D(sA)
        # Expose the K-tile mode required by the MMA B-fragment contract.
        sB = cute.local_tile(
            panel_b,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(None, 1, 1),
        )
        tSsB = thr_mma.partition_B(sB)

        cA = cute.local_tile(
            cute.make_identity_tensor(mA.shape),
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(1, None, 1),
        )
        tAcA = tiled_copy_A.get_slice(tidx).partition_S(cA)
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
                    PACKED_COEFF_DIM,
                )

        panel_k_tiles = TILE_M // self.tile_k
        if cutlass.const_expr(grid_tile == GRID_TILES - 1):
            panel_k_tiles = (GRID_SIZE - TILE_M * grid_tile) // self.tile_k
        panel_k_start = grid_tile * (TILE_M // self.tile_k)
        gmem_pipe_read = cutlass.Int32(panel_k_start)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1
        for stage in range(1, STAGES - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tSsB[None, None, None, 0])
        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        logical_k_tile = cutlass.Int32(0)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(
                tSsB[None, None, 0, logical_k_tile],
                tCrB[None, None, 0],
            )

        for _ in range(panel_k_tiles):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
                    self.cta_sync_barrier.arrive_and_wait()
                k_block_next = (k_block + 1) % k_block_max
                fragment_k_tile = logical_k_tile
                if k_block_max > 1 and k_block == k_block_max - 1:
                    fragment_k_tile = (
                        logical_k_tile + 1
                        if logical_k_tile + 1 < panel_k_tiles
                        else logical_k_tile
                    )
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tSsB[
                        None,
                        None,
                        k_block_next,
                        fragment_k_tile,
                    ],
                    tCrB[None, None, k_block_next],
                )
                if k_block == 0 and tiles_issued < panel_k_tiles:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                cute.gemm(
                    tiled_mma,
                    tCrOut,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrOut,
                )
                if k_block == 0:
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == STAGES:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < panel_k_start + panel_k_tiles
                        else cutlass.Int32(panel_k_start)
                    )
            logical_k_tile = logical_k_tile + 1

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _single_projection_to_shared(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mGrid: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        grid_tile: cutlass.Constexpr,
        channel_tile: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        gA = cute.local_tile(
            mA,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(None, 1, 1),
        )
        gGrid = cute.local_tile(
            mGrid,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, 1, None),
        )
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        if cutlass.const_expr(self.has_channel_residue):
            cB = cute.local_tile(
                cute.make_identity_tensor(mB.shape),
                tiler=self.cta_tiler,
                coord=(0, channel_tile, None),
                proj=(None, 1, 1),
            )
            tBcB = thr_copy_B.partition_S(cB)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB.shape[0][1],
                        cute.size(tBsB, mode=[1]),
                        cute.size(tBsB, mode=[2]),
                    ),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tBpB.shape[0]):
                for channel in range(tBpB.shape[1]):
                    tBpB[rest_v, channel, 0] = cute.elem_less(
                        tBcB[(0, rest_v), channel, 0, 0][0],
                        mB.shape[0],
                    )

        cA = cute.local_tile(
            cute.make_identity_tensor(mA.shape),
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
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

        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA,
        )
        if cutlass.const_expr(self.has_channel_residue):
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, 0],
                pred=tBpB,
            )
        else:
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, 0],
            )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1
        for stage in range(1, STAGES - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            if cutlass.const_expr(self.has_channel_residue):
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, stage],
                    pred=tBpB,
                )
            else:
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, stage],
                )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgGrid = thr_mma.partition_C(gGrid)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrGrid = tiled_mma.make_fragment_C(tCgGrid)
        tCrGrid.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
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
                if k_block == 0 and tiles_issued < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                cute.gemm(
                    tiled_mma,
                    tCrGrid,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrGrid,
                )
                if k_block == 0:
                    if tiles_issued < k_tile_count:
                        if cutlass.const_expr(self.has_channel_residue):
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, gmem_pipe_read],
                                tBsB[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        else:
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, gmem_pipe_read],
                                tBsB[None, None, None, smem_pipe_write],
                            )
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == STAGES:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(0)
                    )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        cGrid = cute.make_identity_tensor(gGrid.shape)
        tCpGrid = thr_mma.partition_C(cGrid)
        pred = cute.make_rmem_tensor(tCrGrid.layout, cutlass.Boolean)
        residue_m = GRID_SIZE - TILE_M * grid_tile
        for idx in range(cute.size(tCrGrid.shape)):
            pred[idx] = cute.elem_less(
                tCpGrid[idx],
                (residue_m, self.cta_tiler[1]),
            )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mGrid.element_type,
        )
        cute.copy(atom, tCrGrid, tCgGrid, pred=pred)
        cute.arch.sync_threads()

    @cute.jit
    def _dual_projection_adjoint(
        self,
        mA: cute.Tensor,
        mB_left: cute.Tensor,
        mB_right: cute.Tensor,
        mGrad_left_grid: cute.Tensor,
        mGrad_right_grid: cute.Tensor,
        sA: cute.Tensor,
        sB_left: cute.Tensor,
        sB_right: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        grid_tile: cutlass.Constexpr,
        channel_tile: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        gA = cute.local_tile(
            mA,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, None, 1),
        )
        gB_left = cute.local_tile(
            mB_left,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(None, 1, 1),
        )
        gB_right = cute.local_tile(
            mB_right,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(None, 1, 1),
        )
        gGrad_left = cute.local_tile(
            mGrad_left_grid,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, 1, None),
        )
        gGrad_right = cute.local_tile(
            mGrad_right_grid,
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
            proj=(1, 1, None),
        )

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB_left = thr_copy_B.partition_S(gB_left)
        tBgB_right = thr_copy_B.partition_S(gB_right)
        tBsB_left = thr_copy_B.partition_D(sB_left)
        tBsB_right = thr_copy_B.partition_D(sB_right)

        if cutlass.const_expr(self.has_channel_residue):
            cB = cute.local_tile(
                cute.make_identity_tensor(mB_left.shape),
                tiler=self.cta_tiler,
                coord=(0, channel_tile, None),
                proj=(None, 1, 1),
            )
            tBcB = thr_copy_B.partition_S(cB)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB_left.shape[0][1],
                        cute.size(tBsB_left, mode=[1]),
                        cute.size(tBsB_left, mode=[2]),
                    ),
                    stride=(cute.size(tBsB_left, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tBpB.shape[0]):
                for channel in range(tBpB.shape[1]):
                    tBpB[rest_v, channel, 0] = cute.elem_less(
                        tBcB[(0, rest_v), channel, 0, 0][0],
                        mB_left.shape[0],
                    )

        cA = cute.local_tile(
            cute.make_identity_tensor(mA.shape),
            tiler=self.cta_tiler,
            coord=(grid_tile, 0, None),
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

        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA,
        )
        if cutlass.const_expr(self.has_channel_residue):
            cute.copy(
                tiled_copy_B,
                tBgB_left[None, None, None, gmem_pipe_read],
                tBsB_left[None, None, None, 0],
                pred=tBpB,
            )
        else:
            cute.copy(
                tiled_copy_B,
                tBgB_left[None, None, None, gmem_pipe_read],
                tBsB_left[None, None, None, 0],
            )
        if cutlass.const_expr(self.has_channel_residue):
            cute.copy(
                tiled_copy_B,
                tBgB_right[None, None, None, gmem_pipe_read],
                tBsB_right[None, None, None, 0],
                pred=tBpB,
            )
        else:
            cute.copy(
                tiled_copy_B,
                tBgB_right[None, None, None, gmem_pipe_read],
                tBsB_right[None, None, None, 0],
            )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1
        for stage in range(1, STAGES - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            if cutlass.const_expr(self.has_channel_residue):
                cute.copy(
                    tiled_copy_B,
                    tBgB_left[None, None, None, gmem_pipe_read],
                    tBsB_left[None, None, None, stage],
                    pred=tBpB,
                )
            else:
                cute.copy(
                    tiled_copy_B,
                    tBgB_left[None, None, None, gmem_pipe_read],
                    tBsB_left[None, None, None, stage],
                )
            if cutlass.const_expr(self.has_channel_residue):
                cute.copy(
                    tiled_copy_B,
                    tBgB_right[None, None, None, gmem_pipe_read],
                    tBsB_right[None, None, None, stage],
                    pred=tBpB,
                )
            else:
                cute.copy(
                    tiled_copy_B,
                    tBgB_right[None, None, None, gmem_pipe_read],
                    tBsB_right[None, None, None, stage],
                )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA = thr_mma.partition_A(sA)
        tCsB_left = thr_mma.partition_B(sB_left)
        tCsB_right = thr_mma.partition_B(sB_right)
        tCgGrad_left = thr_mma.partition_C(gGrad_left)
        tCgGrad_right = thr_mma.partition_C(gGrad_right)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB_left = tiled_mma.make_fragment_B(tCsB_left[None, None, None, 0])
        tCrB_right = tiled_mma.make_fragment_B(tCsB_right[None, None, None, 0])
        tCrLeft = tiled_mma.make_fragment_C(tCgGrad_left)
        tCrRight = tiled_mma.make_fragment_C(tCgGrad_right)
        tCrLeft.fill(0.0)
        tCrRight.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_left_p = tCsB_left[None, None, None, smem_pipe_read]
        tCsB_right_p = tCsB_right[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(
                tCsB_left_p[None, None, 0],
                tCrB_left[None, None, 0],
            )
            cute.autovec_copy(
                tCsB_right_p[None, None, 0],
                tCrB_right[None, None, 0],
            )

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_left_p = tCsB_left[None, None, None, smem_pipe_read]
                    tCsB_right_p = tCsB_right[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
                    self.cta_sync_barrier.arrive_and_wait()
                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_left_p[None, None, k_block_next],
                    tCrB_left[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_right_p[None, None, k_block_next],
                    tCrB_right[None, None, k_block_next],
                )
                if k_block == 0 and tiles_issued < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                cute.gemm(
                    tiled_mma,
                    tCrLeft,
                    tCrA[None, None, k_block],
                    tCrB_left[None, None, k_block],
                    tCrLeft,
                )
                cute.gemm(
                    tiled_mma,
                    tCrRight,
                    tCrA[None, None, k_block],
                    tCrB_right[None, None, k_block],
                    tCrRight,
                )
                if k_block == 0:
                    if tiles_issued < k_tile_count:
                        if cutlass.const_expr(self.has_channel_residue):
                            cute.copy(
                                tiled_copy_B,
                                tBgB_left[None, None, None, gmem_pipe_read],
                                tBsB_left[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        else:
                            cute.copy(
                                tiled_copy_B,
                                tBgB_left[None, None, None, gmem_pipe_read],
                                tBsB_left[None, None, None, smem_pipe_write],
                            )
                        if cutlass.const_expr(self.has_channel_residue):
                            cute.copy(
                                tiled_copy_B,
                                tBgB_right[None, None, None, gmem_pipe_read],
                                tBsB_right[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        else:
                            cute.copy(
                                tiled_copy_B,
                                tBgB_right[None, None, None, gmem_pipe_read],
                                tBsB_right[None, None, None, smem_pipe_write],
                            )
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == STAGES:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(0)
                    )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        cGrid = cute.make_identity_tensor(gGrad_left.shape)
        tCpGrid = thr_mma.partition_C(cGrid)
        pred = cute.make_rmem_tensor(tCrLeft.layout, cutlass.Boolean)
        tCrDP = tiled_mma.make_fragment_C(tCgGrad_left)
        tCrDP.fill(0.0)
        residue_m = GRID_SIZE - TILE_M * grid_tile
        for idx in range(cute.size(tCrLeft.shape)):
            pred[idx] = cute.elem_less(
                tCpGrid[idx],
                (residue_m, self.cta_tiler[1]),
            )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mGrad_left_grid.element_type,
        )
        cute.copy(atom, tCgGrad_left, tCrDP, pred=pred)
        for idx in range(cute.size(tCrLeft.shape)):
            if pred[idx]:
                left_value = tCrLeft[idx].to(cutlass.Float32)
                right_value = tCrRight[idx].to(cutlass.Float32)
                grad_product = tCrDP[idx].to(cutlass.Float32)
                tCrLeft[idx] = grad_product * right_value
                tCrRight[idx] = grad_product * left_value
        cute.copy(atom, tCrLeft, tCgGrad_left, pred=pred)
        cute.copy(atom, tCrRight, tCgGrad_right, pred=pred)
        cute.arch.sync_threads()

    @cute.jit
    def _dual_backproject(
        self,
        mA: cute.Tensor,
        mB_left_shared: cute.Tensor,
        mB_right_shared: cute.Tensor,
        mOut_left: cute.Tensor,
        mOut_right: cute.Tensor,
        sA: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        channel_tile: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        gA = cute.local_tile(
            mA,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(1, None, 1),
        )
        sB_left = cute.local_tile(
            mB_left_shared,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(None, 1, 1),
        )
        sB_right = cute.local_tile(
            mB_right_shared,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(None, 1, 1),
        )
        gOut_left = cute.local_tile(
            mOut_left,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(1, 1, None),
        )
        gOut_right = cute.local_tile(
            mOut_right,
            tiler=self.cta_tiler,
            coord=(0, channel_tile, None),
            proj=(1, 1, None),
        )
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)

        cA = cute.local_tile(
            cute.make_identity_tensor(mA.shape),
            tiler=self.cta_tiler,
            coord=(0, 0, None),
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
                    PACKED_COEFF_DIM,
                )

        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1
        for stage in range(1, STAGES - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA = thr_mma.partition_A(sA)
        tSsB_left = thr_mma.partition_B(sB_left)
        tSsB_right = thr_mma.partition_B(sB_right)
        tCgOut_left = thr_mma.partition_C(gOut_left)
        tCgOut_right = thr_mma.partition_C(gOut_right)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB_left = tiled_mma.make_fragment_B(tSsB_left[None, None, None, 0])
        tCrB_right = tiled_mma.make_fragment_B(tSsB_right[None, None, None, 0])
        tCrOut_left = tiled_mma.make_fragment_C(tCgOut_left)
        tCrOut_right = tiled_mma.make_fragment_C(tCgOut_right)
        tCrOut_left.fill(0.0)
        tCrOut_right.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        logical_k_tile = cutlass.Int32(0)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(
                tSsB_left[None, None, 0, logical_k_tile],
                tCrB_left[None, None, 0],
            )
            cute.autovec_copy(
                tSsB_right[None, None, 0, logical_k_tile],
                tCrB_right[None, None, 0],
            )

        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
                    self.cta_sync_barrier.arrive_and_wait()
                k_block_next = (k_block + 1) % k_block_max
                fragment_k_tile = logical_k_tile
                if k_block_max > 1:
                    if k_block == k_block_max - 1:
                        fragment_k_tile = (
                            logical_k_tile + 1
                            if logical_k_tile + 1 < k_tile_count
                            else logical_k_tile
                        )
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tSsB_left[
                        None,
                        None,
                        k_block_next,
                        fragment_k_tile,
                    ],
                    tCrB_left[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tSsB_right[
                        None,
                        None,
                        k_block_next,
                        fragment_k_tile,
                    ],
                    tCrB_right[None, None, k_block_next],
                )
                if k_block == 0 and tiles_issued < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                cute.gemm(
                    tiled_mma,
                    tCrOut_left,
                    tCrA[None, None, k_block],
                    tCrB_left[None, None, k_block],
                    tCrOut_left,
                )
                cute.gemm(
                    tiled_mma,
                    tCrOut_right,
                    tCrA[None, None, k_block],
                    tCrB_right[None, None, k_block],
                    tCrOut_right,
                )
                if k_block == 0:
                    cute.arch.cp_async_commit_group()
                    tiles_issued = tiles_issued + 1
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == STAGES:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(0)
                    )
            logical_k_tile = logical_k_tile + 1

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        pred = cute.make_rmem_tensor(tCrOut_left.layout, cutlass.Boolean)
        if cutlass.const_expr(self.has_channel_residue):
            cOut = cute.local_tile(
                cute.make_identity_tensor(mOut_left.shape),
                tiler=self.cta_tiler,
                coord=(0, channel_tile, None),
                proj=(1, 1, None),
            )
            tCpOut = thr_mma.partition_C(cOut)
            for idx in range(cute.size(tCrOut_left.shape)):
                pred[idx] = cute.elem_less(tCpOut[idx], mOut_left.shape)
        else:
            cOut = cute.make_identity_tensor(gOut_left.shape)
            tCpOut = thr_mma.partition_C(cOut)
            for idx in range(cute.size(tCrOut_left.shape)):
                pred[idx] = cute.elem_less(
                    tCpOut[idx],
                    (PACKED_COEFF_DIM, self.cta_tiler[1]),
                )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mOut_left.element_type,
        )
        cute.copy(atom, tCrOut_left, tCgOut_left, pred=pred)
        cute.copy(atom, tCrOut_right, tCgOut_right, pred=pred)


def compile_tiled_output_grid_product(
    device_index: int | None = None,
    compute_capability: tuple[int, int] | None = None,
    hidden_channels: int = 192,
    tile_n: int = TILE_N,
    channel_tile_start: int = 0,
    channel_tile_count: int | None = None,
) -> Callable:
    """Compile one forward artifact with symbolic runtime node count."""
    import torch

    hidden_channels = _validate_hidden_channels(hidden_channels)
    tile_n = int(tile_n)
    if tile_n not in (C96_TAIL_TILE_N, SM80_C96_TILE_N, TILE_N):
        raise ValueError("output-grid forward tile_n must be 32, 48, or 64")
    if device_index is None:
        device_index = torch.cuda.current_device()
    actual_capability = tuple(torch.cuda.get_device_capability(device_index))
    if (
        compute_capability is not None
        and tuple(compute_capability) != actual_capability
    ):
        raise ValueError("compile target does not match the selected CUDA device")
    if tile_n == SM80_C96_TILE_N and (
        actual_capability not in runtime_policy.SM80_PROFILE_CAPABILITIES
        or hidden_channels != 96
    ):
        raise ValueError("output-grid forward N=48 requires SM80-family and C=96")
    if tile_n == C96_TAIL_TILE_N and (
        actual_capability != (9, 0)
        or hidden_channels != 96
        or int(channel_tile_start) != C96_TAIL_CHANNEL_TILE
        or int(channel_tile_count or 0) != 1
    ):
        raise ValueError("output-grid forward N=32 tail requires sm90 and C=96")
    if (
        tile_n == TILE_N
        and channel_tile_count is not None
        and (
            actual_capability != (9, 0)
            or hidden_channels != 96
            or int(channel_tile_start) != 0
            or int(channel_tile_count) != 1
        )
    ):
        raise ValueError(
            "partial output-grid forward N=64 launch requires sm90 C=96 base panel"
        )
    with torch.cuda.device(device_index):
        nodes = cute.sym_int64()
        fake_coeff = make_fake_compact_tensor(
            cutlass.Float32,
            (nodes, PACKED_COEFF_DIM, hidden_channels),
            stride_order=(2, 1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_to_grid = make_fake_compact_tensor(
            cutlass.Float32,
            (GRID_SIZE, PACKED_COEFF_DIM),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_from_grid = make_fake_compact_tensor(
            cutlass.Float32,
            (PACKED_COEFF_DIM, GRID_SIZE),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            TiledOutputGridProductForward(
                hidden_channels,
                tile_n=tile_n,
                channel_tile_start=channel_tile_start,
                channel_tile_count=channel_tile_count,
            ),
            fake_coeff,
            fake_coeff,
            fake_to_grid,
            fake_from_grid,
            fake_coeff,
            fake_stream,
            options="--enable-tvm-ffi",
        )


def compile_tiled_output_grid_product_backward(
    device_index: int | None = None,
    compute_capability: tuple[int, int] | None = None,
    hidden_channels: int = 192,
    tile_n: int = TILE_N,
    channel_tile_start: int = 0,
    channel_tile_count: int | None = None,
) -> Callable:
    """Compile one first-backward artifact with symbolic node count."""
    import torch

    hidden_channels = _validate_hidden_channels(hidden_channels)
    tile_n = int(tile_n)
    if tile_n not in (C96_TAIL_TILE_N, SM80_C96_TILE_N, TILE_N):
        raise ValueError("output-grid backward tile_n must be 32, 48, or 64")
    if device_index is None:
        device_index = torch.cuda.current_device()
    actual_capability = tuple(torch.cuda.get_device_capability(device_index))
    if (
        compute_capability is not None
        and tuple(compute_capability) != actual_capability
    ):
        raise ValueError("compile target does not match the selected CUDA device")
    if tile_n == SM80_C96_TILE_N and (
        actual_capability not in runtime_policy.SM80_PROFILE_CAPABILITIES
        or hidden_channels != 96
    ):
        raise ValueError(
            "output-grid N=48 panel adjoint requires SM80-family, C=96, and K=8"
        )
    if tile_n == C96_TAIL_TILE_N and (
        actual_capability != (9, 0)
        or hidden_channels != 96
        or int(channel_tile_start) != C96_TAIL_CHANNEL_TILE
        or int(channel_tile_count or 0) != 1
    ):
        raise ValueError("output-grid backward N=32 tail requires sm90, C=96, and K=8")
    if (
        tile_n == TILE_N
        and channel_tile_count is not None
        and (
            actual_capability != (9, 0)
            or hidden_channels != 96
            or int(channel_tile_start) != 0
            or int(channel_tile_count) != 1
        )
    ):
        raise ValueError(
            "partial output-grid backward N=64 launch requires sm90 C=96 K=8 base panel"
        )
    with torch.cuda.device(device_index):
        nodes = cute.sym_int64()
        fake_coeff = make_fake_compact_tensor(
            cutlass.Float32,
            (nodes, PACKED_COEFF_DIM, hidden_channels),
            stride_order=(2, 1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_to_grid = make_fake_compact_tensor(
            cutlass.Float32,
            (GRID_SIZE, PACKED_COEFF_DIM),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_from_grid = make_fake_compact_tensor(
            cutlass.Float32,
            (PACKED_COEFF_DIM, GRID_SIZE),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            TiledOutputGridProductBackward(
                hidden_channels,
                tile_n=tile_n,
                channel_tile_start=channel_tile_start,
                channel_tile_count=channel_tile_count,
            ),
            fake_coeff,
            fake_coeff,
            fake_coeff,
            fake_to_grid,
            fake_from_grid,
            fake_coeff,
            fake_coeff,
            fake_stream,
            options="--enable-tvm-ffi",
        )


@lru_cache(maxsize=16)
def _compiled_tiled_forward(
    device_index: int,
    compute_capability: tuple[int, int],
    hidden_channels: int,
    tile_n: int,
    channel_tile_start: int,
    channel_tile_count: int | None,
) -> Callable:
    return compile_tiled_output_grid_product(
        device_index,
        compute_capability,
        hidden_channels,
        tile_n,
        channel_tile_start,
        channel_tile_count,
    )


@lru_cache(maxsize=16)
def _compiled_tiled_backward(
    device_index: int,
    compute_capability: tuple[int, int],
    hidden_channels: int,
    tile_n: int,
    channel_tile_start: int,
    channel_tile_count: int | None,
) -> Callable:
    return compile_tiled_output_grid_product_backward(
        device_index,
        compute_capability,
        hidden_channels,
        tile_n,
        channel_tile_start,
        channel_tile_count,
    )


def _compile_key(tensor) -> tuple[int, tuple[int, int], int]:
    import torch

    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compute_capability = tuple(torch.cuda.get_device_capability(device_index))
    return int(device_index), compute_capability, int(tensor.shape[2])


def _validate_tensors(left, right, to_grid, from_grid, out=None) -> None:
    import torch

    tensors = (left, right, to_grid, from_grid)
    if (
        any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.dtype != torch.float32 for tensor in tensors)
        or any(tensor.device != left.device for tensor in tensors)
        or left.ndim != 3
        or left.shape[0] <= 0
        or int(left.shape[1]) != PACKED_COEFF_DIM
        or int(left.shape[2]) not in SUPPORTED_HIDDEN_CHANNELS
        or right.shape != left.shape
        or tuple(to_grid.shape) != (GRID_SIZE, PACKED_COEFF_DIM)
        or tuple(from_grid.shape) != (PACKED_COEFF_DIM, GRID_SIZE)
        or any(not tensor.is_contiguous() for tensor in tensors)
        or torch.cuda.get_device_capability(left.device)[0] < 8
    ):
        raise ValueError(
            "tiled output-grid product requires contiguous CUDA FP32 "
            "left/right=(N,48,C) with C in {96,192}, to_grid=(152,48), and "
            "from_grid=(48,152) tensors on compute capability 8.0+"
        )
    if out is not None and (
        out.shape != left.shape
        or out.device != left.device
        or out.dtype != left.dtype
        or not out.is_contiguous()
    ):
        raise ValueError("tiled output-grid output must match left")


def run_tiled_output_grid_product(
    left,
    right,
    to_grid,
    from_grid,
    *,
    use_sm80_c96_n48: bool = False,
    use_sm90_c96_asymmetric_panels: bool = False,
):
    """Run the fused strict-FP32 tiled output-grid forward."""
    import torch

    _validate_tensors(left, right, to_grid, from_grid)
    out = torch.empty_like(left)
    return run_tiled_output_grid_product_out(
        left,
        right,
        to_grid,
        from_grid,
        out,
        use_sm80_c96_n48=use_sm80_c96_n48,
        use_sm90_c96_asymmetric_panels=use_sm90_c96_asymmetric_panels,
    )


def run_tiled_output_grid_product_out(
    left,
    right,
    to_grid,
    from_grid,
    out,
    *,
    use_sm80_c96_n48: bool = False,
    use_sm90_c96_asymmetric_panels: bool = False,
):
    """Run the fused forward into a caller-provided output tensor."""
    _validate_tensors(left, right, to_grid, from_grid, out)
    compile_key = _compile_key(left)
    if use_sm80_c96_n48 and use_sm90_c96_asymmetric_panels:
        raise ValueError("output-grid panel specializations are mutually exclusive")
    if use_sm80_c96_n48 and (
        compile_key[1] not in runtime_policy.SM80_PROFILE_CAPABILITIES
        or compile_key[2] != 96
    ):
        raise ValueError("output-grid forward N=48 requires SM80-family and C=96")
    if use_sm90_c96_asymmetric_panels and (
        compile_key[1] != (9, 0) or compile_key[2] != 96
    ):
        raise ValueError("output-grid asymmetric forward requires sm90 and C=96")
    if use_sm90_c96_asymmetric_panels:
        _compiled_tiled_forward(
            *compile_key,
            TILE_N,
            0,
            1,
        )(
            left,
            right,
            to_grid,
            from_grid,
            out,
        )
        _compiled_tiled_forward(
            *compile_key,
            C96_TAIL_TILE_N,
            C96_TAIL_CHANNEL_TILE,
            1,
        )(
            left,
            right,
            to_grid,
            from_grid,
            out,
        )
        return out
    tile_n = SM80_C96_TILE_N if use_sm80_c96_n48 else TILE_N
    _compiled_tiled_forward(*compile_key, tile_n, 0, None)(
        left,
        right,
        to_grid,
        from_grid,
        out,
    )
    return out


def run_tiled_output_grid_product_backward(
    grad_out,
    left,
    right,
    to_grid,
    from_grid,
    *,
    use_sm80_c96_n48_panel: bool = False,
    use_sm90_c96_asymmetric_panels: bool = False,
):
    """Run the complete fused first backward for left and right inputs."""
    import torch

    _validate_tensors(left, right, to_grid, from_grid)
    if (
        grad_out.shape != left.shape
        or grad_out.device != left.device
        or grad_out.dtype != left.dtype
        or not grad_out.is_contiguous()
    ):
        raise ValueError("grad_out must be contiguous and match left")
    compile_key = _compile_key(left)
    if use_sm80_c96_n48_panel and use_sm90_c96_asymmetric_panels:
        raise ValueError("output-grid panel specializations are mutually exclusive")
    if use_sm80_c96_n48_panel and (
        compile_key[1] not in runtime_policy.SM80_PROFILE_CAPABILITIES
        or compile_key[2] != 96
    ):
        raise ValueError(
            "output-grid N=48 panel adjoint requires SM80-family, C=96, and K=8"
        )
    if use_sm90_c96_asymmetric_panels and (
        compile_key[1] != (9, 0) or compile_key[2] != 96
    ):
        raise ValueError("output-grid asymmetric backward requires sm90, C=96, and K=8")
    grad_left = torch.empty_like(left)
    grad_right = torch.empty_like(right)
    if use_sm90_c96_asymmetric_panels:
        _compiled_tiled_backward(
            *compile_key,
            TILE_N,
            0,
            1,
        )(
            grad_out,
            left,
            right,
            to_grid,
            from_grid,
            grad_left,
            grad_right,
        )
        _compiled_tiled_backward(
            *compile_key,
            C96_TAIL_TILE_N,
            C96_TAIL_CHANNEL_TILE,
            1,
        )(
            grad_out,
            left,
            right,
            to_grid,
            from_grid,
            grad_left,
            grad_right,
        )
        return grad_left, grad_right
    tile_n = SM80_C96_TILE_N if use_sm80_c96_n48_panel else TILE_N
    _compiled_tiled_backward(
        *compile_key,
        tile_n,
        0,
        None,
    )(
        grad_out,
        left,
        right,
        to_grid,
        from_grid,
        grad_left,
        grad_right,
    )
    return grad_left, grad_right


__all__ = [
    "TiledOutputGridProductBackward",
    "TiledOutputGridProductForward",
    "run_tiled_output_grid_product",
    "run_tiled_output_grid_product_backward",
    "run_tiled_output_grid_product_out",
]
