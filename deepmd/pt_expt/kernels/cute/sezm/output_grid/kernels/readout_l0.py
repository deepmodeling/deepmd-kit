# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tiled strict-FP32 Neo output readout for degree zero only."""

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

from .tiled_product import (
    FAKE_TENSOR_KW,
    GRID_SIZE,
    PACKED_COEFF_DIM,
    STAGES,
    THREADS,
    TILE_K,
    TILE_M,
    TILE_N,
)

# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, TC002, TC003


HIDDEN_CHANNELS = 192
GRAM_FORWARD_THREADS = HIDDEN_CHANNELS
GRAM_ELEMENTS = PACKED_COEFF_DIM * PACKED_COEFF_DIM


class TiledReadoutL0GramForward:
    """Evaluate the channelwise Gram bilinear with one CTA per node."""

    @cute.jit
    def __call__(
        self,
        left: cute.Tensor,
        right: cute.Tensor,
        gram: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ):
        gram_layout = cute.make_layout(
            (PACKED_COEFF_DIM, PACKED_COEFF_DIM),
            stride=(PACKED_COEFF_DIM, 1),
        )
        right_layout = cute.make_layout(
            (PACKED_COEFF_DIM, HIDDEN_CHANNELS),
            stride=(HIDDEN_CHANNELS, 1),
        )
        self.kernel(
            left,
            right,
            gram,
            out,
            gram_layout,
            right_layout,
        ).launch(
            grid=(left.shape[0], 1, 1),
            block=[GRAM_FORWARD_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        left: cute.Tensor,
        right: cute.Tensor,
        gram: cute.Tensor,
        out: cute.Tensor,
        gram_layout: cute.Layout,
        right_layout: cute.Layout,
    ):
        channel, _, _ = cute.arch.thread_idx()
        node, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        s_gram = smem.allocate_tensor(cutlass.Float32, gram_layout, 16)
        s_right = smem.allocate_tensor(cutlass.Float32, right_layout, 16)

        for linear in cutlass.range(
            channel,
            GRAM_ELEMENTS,
            GRAM_FORWARD_THREADS,
            unroll=1,
        ):
            row = linear // PACKED_COEFF_DIM
            col = linear - row * PACKED_COEFF_DIM
            s_gram[row, col] = gram[row, col].to(cutlass.Float32)
        for coeff in cutlass.range(0, PACKED_COEFF_DIM, 1, unroll=1):
            s_right[coeff, channel] = right[node, coeff, channel].to(cutlass.Float32)
        cute.arch.sync_threads()

        value = cutlass.Float32(0.0)
        for row in cutlass.range(0, PACKED_COEFF_DIM, 1, unroll=1):
            transformed_right = cutlass.Float32(0.0)
            for col in cutlass.range(0, PACKED_COEFF_DIM, 1, unroll=1):
                transformed_right += s_gram[row, col] * s_right[col, channel]
            value += left[node, row, channel].to(cutlass.Float32) * transformed_right
        out[node, channel] = value.to(out.element_type)


class TiledReadoutL0GramBackward:
    """Apply the frozen 48x48 Gram matrix to one 64-channel tile."""

    def __init__(self) -> None:
        self.cta_tiler = (TILE_M, TILE_N, TILE_K)
        self.channel_tiles = HIDDEN_CHANNELS // TILE_N
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=THREADS,
        )

    @cute.jit
    def __call__(
        self,
        dq0: cute.Tensor,
        left: cute.Tensor,
        right: cute.Tensor,
        gram: cute.Tensor,
        grad_left: cute.Tensor,
        grad_right: cute.Tensor,
        stream: CUstream,
    ):
        sA_layout = cute.make_layout(
            (TILE_M, TILE_K, STAGES),
            stride=(1, TILE_M + 4, TILE_K * (TILE_M + 4)),
        )
        sB_layout = cute.make_layout(
            (TILE_N, TILE_K, STAGES),
            stride=(1, TILE_N, TILE_K * TILE_N),
        )
        dq0_layout = cute.make_layout((TILE_N,), stride=(1,))

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

        vector = 4
        copy_b_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            left.element_type,
            num_bits_per_copy=left.element_type.width * vector,
        )
        copy_b_major = TILE_N // vector
        copy_b_layout = cute.make_layout(
            (copy_b_major, THREADS // copy_b_major),
            stride=(1, copy_b_major),
        )
        tiled_copy_B = cute.make_tiled_copy_tv(
            copy_b_atom,
            copy_b_layout,
            cute.make_layout((vector, 1)),
        )

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
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.MmaUniversalOp(cutlass.Float32),
            atoms_layout,
            permutation_mnk=(permutation_m, permutation_n, None),
        )

        self.kernel(
            dq0,
            left,
            right,
            gram,
            grad_left,
            grad_right,
            sA_layout,
            sB_layout,
            dq0_layout,
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
        dq0: cute.Tensor,
        left: cute.Tensor,
        right: cute.Tensor,
        gram: cute.Tensor,
        grad_left: cute.Tensor,
        grad_right: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        dq0_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        node, channel_tile, _ = cute.arch.block_idx()

        matrix_b_layout = cute.make_layout(
            (HIDDEN_CHANNELS, PACKED_COEFF_DIM),
            stride=(1, HIDDEN_CHANNELS),
        )
        left_b = cute.make_tensor(
            left[node, None, None].iterator,
            matrix_b_layout,
        )
        right_b = cute.make_tensor(
            right[node, None, None].iterator,
            matrix_b_layout,
        )
        gram_t = cute.make_tensor(
            gram.iterator,
            cute.make_layout(
                (PACKED_COEFF_DIM, PACKED_COEFF_DIM),
                stride=(1, PACKED_COEFF_DIM),
            ),
        )

        smem = cutlass.utils.SmemAllocator()
        sA_left = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
        sA_right = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
        sB_left = smem.allocate_tensor(cutlass.Float32, sB_layout, 16)
        sB_right = smem.allocate_tensor(cutlass.Float32, sB_layout, 16)
        sDq0 = smem.allocate_tensor(cutlass.Float32, dq0_layout, 16)

        for local_channel in cutlass.range(
            tidx,
            TILE_N,
            THREADS,
            unroll=1,
        ):
            channel = channel_tile * TILE_N + local_channel
            sDq0[local_channel] = dq0[node, channel].to(cutlass.Float32)
        cute.arch.sync_threads()

        self._dual_gram_adjoint(
            gram,
            gram_t,
            right_b,
            left_b,
            grad_left[node, None, None],
            grad_right[node, None, None],
            sA_left,
            sA_right,
            sB_left,
            sB_right,
            sDq0,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            tidx,
            channel_tile,
        )

    @cute.jit
    def _dual_gram_adjoint(
        self,
        mA_left: cute.Tensor,
        mA_right: cute.Tensor,
        mB_left: cute.Tensor,
        mB_right: cute.Tensor,
        mOut_left: cute.Tensor,
        mOut_right: cute.Tensor,
        sA_left: cute.Tensor,
        sA_right: cute.Tensor,
        sB_left: cute.Tensor,
        sB_right: cute.Tensor,
        sDq0: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        channel_tile: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        gA_left = cute.local_tile(
            mA_left,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(1, None, 1),
        )
        gA_right = cute.local_tile(
            mA_right,
            tiler=self.cta_tiler,
            coord=(0, 0, None),
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
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA_left = thr_copy_A.partition_S(gA_left)
        tAgA_right = thr_copy_A.partition_S(gA_right)
        tAsA_left = thr_copy_A.partition_D(sA_left)
        tAsA_right = thr_copy_A.partition_D(sA_right)
        tBgB_left = thr_copy_B.partition_S(gB_left)
        tBgB_right = thr_copy_B.partition_S(gB_right)
        tBsB_left = thr_copy_B.partition_D(sB_left)
        tBsB_right = thr_copy_B.partition_D(sB_right)

        cA = cute.local_tile(
            cute.make_identity_tensor(mA_left.shape),
            tiler=self.cta_tiler,
            coord=(0, 0, None),
            proj=(1, None, 1),
        )
        tAcA = thr_copy_A.partition_S(cA)
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAsA_left.shape[0][1],
                    cute.size(tAsA_left, mode=[1]),
                    cute.size(tAsA_left, mode=[2]),
                ),
                stride=(cute.size(tAsA_left, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for row in range(tApA.shape[1]):
                tApA[rest_v, row, 0] = cute.elem_less(
                    tAcA[(0, rest_v), row, 0, 0][0],
                    PACKED_COEFF_DIM,
                )

        k_tile_count = cute.size(tAgA_left, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA_left[None, None, None, gmem_pipe_read],
            tAsA_left[None, None, None, 0],
            pred=tApA,
        )
        cute.copy(
            tiled_copy_A,
            tAgA_right[None, None, None, gmem_pipe_read],
            tAsA_right[None, None, None, 0],
            pred=tApA,
        )
        cute.copy(
            tiled_copy_B,
            tBgB_left[None, None, None, gmem_pipe_read],
            tBsB_left[None, None, None, 0],
        )
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
                tAgA_left[None, None, None, gmem_pipe_read],
                tAsA_left[None, None, None, stage],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_A,
                tAgA_right[None, None, None, gmem_pipe_read],
                tAsA_right[None, None, None, stage],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB_left[None, None, None, gmem_pipe_read],
                tBsB_left[None, None, None, stage],
            )
            cute.copy(
                tiled_copy_B,
                tBgB_right[None, None, None, gmem_pipe_read],
                tBsB_right[None, None, None, stage],
            )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = gmem_pipe_read + 1

        tCsA_left = thr_mma.partition_A(sA_left)
        tCsA_right = thr_mma.partition_A(sA_right)
        tCsB_left = thr_mma.partition_B(sB_left)
        tCsB_right = thr_mma.partition_B(sB_right)
        tCgOut_left = thr_mma.partition_C(gOut_left)
        tCgOut_right = thr_mma.partition_C(gOut_right)
        tCrA_left = tiled_mma.make_fragment_A(tCsA_left[None, None, None, 0])
        tCrA_right = tiled_mma.make_fragment_A(tCsA_right[None, None, None, 0])
        tCrB_left = tiled_mma.make_fragment_B(tCsB_left[None, None, None, 0])
        tCrB_right = tiled_mma.make_fragment_B(tCsB_right[None, None, None, 0])
        tCrOut_left = tiled_mma.make_fragment_C(tCgOut_left)
        tCrOut_right = tiled_mma.make_fragment_C(tCgOut_right)
        tCrOut_left.fill(0.0)
        tCrOut_right.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(STAGES - 1)
        tiles_issued = cutlass.Int32(STAGES - 1)
        tCsA_left_p = tCsA_left[None, None, None, smem_pipe_read]
        tCsA_right_p = tCsA_right[None, None, None, smem_pipe_read]
        tCsB_left_p = tCsB_left[None, None, None, smem_pipe_read]
        tCsB_right_p = tCsB_right[None, None, None, smem_pipe_read]
        k_block_max = cute.size(tCrA_left, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(STAGES - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(
                tCsA_left_p[None, None, 0],
                tCrA_left[None, None, 0],
            )
            cute.autovec_copy(
                tCsA_right_p[None, None, 0],
                tCrA_right[None, None, 0],
            )
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
                    tCsA_left_p = tCsA_left[None, None, None, smem_pipe_read]
                    tCsA_right_p = tCsA_right[None, None, None, smem_pipe_read]
                    tCsB_left_p = tCsB_left[None, None, None, smem_pipe_read]
                    tCsB_right_p = tCsB_right[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(STAGES - 2)
                    self.cta_sync_barrier.arrive_and_wait()
                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(
                    tCsA_left_p[None, None, k_block_next],
                    tCrA_left[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsA_right_p[None, None, k_block_next],
                    tCrA_right[None, None, k_block_next],
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
                        tAgA_left[None, None, None, gmem_pipe_read],
                        tAsA_left[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                    cute.copy(
                        tiled_copy_A,
                        tAgA_right[None, None, None, gmem_pipe_read],
                        tAsA_right[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBgB_left[None, None, None, gmem_pipe_read],
                        tBsB_left[None, None, None, smem_pipe_write],
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBgB_right[None, None, None, gmem_pipe_read],
                        tBsB_right[None, None, None, smem_pipe_write],
                    )
                cute.gemm(
                    tiled_mma,
                    tCrOut_left,
                    tCrA_left[None, None, k_block],
                    tCrB_left[None, None, k_block],
                    tCrOut_left,
                )
                cute.gemm(
                    tiled_mma,
                    tCrOut_right,
                    tCrA_right[None, None, k_block],
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

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        cOut = cute.make_identity_tensor(gOut_left.shape)
        tCpOut = thr_mma.partition_C(cOut)
        pred = cute.make_rmem_tensor(tCrOut_left.layout, cutlass.Boolean)
        for idx in range(cute.size(tCrOut_left.shape)):
            pred[idx] = cute.elem_less(
                tCpOut[idx],
                (PACKED_COEFF_DIM, TILE_N),
            )
            if pred[idx]:
                local_channel = tCpOut[idx][1]
                scale = sDq0[local_channel].to(cutlass.Float32)
                tCrOut_left[idx] = tCrOut_left[idx].to(cutlass.Float32) * scale
                tCrOut_right[idx] = tCrOut_right[idx].to(cutlass.Float32) * scale
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mOut_left.element_type,
        )
        cute.copy(atom, tCrOut_left, tCgOut_left, pred=pred)
        cute.copy(atom, tCrOut_right, tCgOut_right, pred=pred)


def compile_readout_l0_gram_forward(
    device_index: int | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> Callable:
    """Compile the dense-Gram forward with symbolic runtime node count."""
    import torch

    if device_index is None:
        device_index = torch.cuda.current_device()
    actual_capability = tuple(torch.cuda.get_device_capability(device_index))
    if (
        compute_capability is not None
        and tuple(compute_capability) != actual_capability
    ):
        raise ValueError("compile target does not match the selected CUDA device")
    with torch.cuda.device(device_index):
        nodes = cute.sym_int64()
        fake_coeff = make_fake_compact_tensor(
            cutlass.Float32,
            (nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS),
            stride_order=(2, 1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_gram = make_fake_compact_tensor(
            cutlass.Float32,
            (PACKED_COEFF_DIM, PACKED_COEFF_DIM),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_out = make_fake_compact_tensor(
            cutlass.Float32,
            (nodes, HIDDEN_CHANNELS),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            TiledReadoutL0GramForward(),
            fake_coeff,
            fake_coeff,
            fake_gram,
            fake_out,
            fake_stream,
            options="--enable-tvm-ffi",
        )


def compile_readout_l0_gram_backward(
    device_index: int | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> Callable:
    """Compile the dense-Gram first-backward artifact."""
    import torch

    if device_index is None:
        device_index = torch.cuda.current_device()
    actual_capability = tuple(torch.cuda.get_device_capability(device_index))
    if (
        compute_capability is not None
        and tuple(compute_capability) != actual_capability
    ):
        raise ValueError("compile target does not match the selected CUDA device")
    with torch.cuda.device(device_index):
        nodes = cute.sym_int64()
        fake_coeff = make_fake_compact_tensor(
            cutlass.Float32,
            (nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS),
            stride_order=(2, 1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_q0 = make_fake_compact_tensor(
            cutlass.Float32,
            (nodes, HIDDEN_CHANNELS),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_gram = make_fake_compact_tensor(
            cutlass.Float32,
            (PACKED_COEFF_DIM, PACKED_COEFF_DIM),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            TiledReadoutL0GramBackward(),
            fake_q0,
            fake_coeff,
            fake_coeff,
            fake_gram,
            fake_coeff,
            fake_coeff,
            fake_stream,
            options="--enable-tvm-ffi",
        )


@lru_cache(maxsize=16)
def _compiled_readout_l0_gram_forward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    return compile_readout_l0_gram_forward(device_index, compute_capability)


@lru_cache(maxsize=16)
def _compiled_readout_l0_gram_backward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    return compile_readout_l0_gram_backward(device_index, compute_capability)


def _compile_key(tensor) -> tuple[int, tuple[int, int]]:
    import torch

    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compute_capability = tuple(torch.cuda.get_device_capability(device_index))
    return int(device_index), compute_capability


def _validate_tensors(left, right, to_grid, from_grid, out=None) -> None:
    import torch

    tensors = (left, right, to_grid, from_grid)
    if (
        any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.dtype != torch.float32 for tensor in tensors)
        or any(tensor.device != left.device for tensor in tensors)
        or left.ndim != 3
        or left.shape[0] <= 0
        or tuple(left.shape[1:]) != (PACKED_COEFF_DIM, HIDDEN_CHANNELS)
        or right.shape != left.shape
        or tuple(to_grid.shape) != (GRID_SIZE, PACKED_COEFF_DIM)
        or tuple(from_grid.shape) != (PACKED_COEFF_DIM, GRID_SIZE)
        or any(not tensor.is_contiguous() for tensor in tensors)
        or any(tensor.data_ptr() % 16 != 0 for tensor in tensors)
        or torch.cuda.get_device_capability(left.device)[0] < 8
    ):
        raise ValueError(
            "readout l=0 requires contiguous CUDA FP32 left/right=(N,48,192), "
            "to_grid=(152,48), and from_grid=(48,152) tensors on compute "
            "capability 8.0+"
        )
    if out is not None and (
        tuple(out.shape) != (left.shape[0], HIDDEN_CHANNELS)
        or out.device != left.device
        or out.dtype != left.dtype
        or not out.is_contiguous()
        or out.data_ptr() % 16 != 0
    ):
        raise ValueError("readout l=0 output must be contiguous with shape (N,192)")


def _validate_gram_tensors(left, right, gram, out=None) -> None:
    import torch

    tensors = (left, right, gram)
    if (
        any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.dtype != torch.float32 for tensor in tensors)
        or any(tensor.device != left.device for tensor in tensors)
        or left.ndim != 3
        or left.shape[0] <= 0
        or tuple(left.shape[1:]) != (PACKED_COEFF_DIM, HIDDEN_CHANNELS)
        or right.shape != left.shape
        or tuple(gram.shape) != (PACKED_COEFF_DIM, PACKED_COEFF_DIM)
        or any(not tensor.is_contiguous() for tensor in tensors)
        or any(tensor.data_ptr() % 16 != 0 for tensor in tensors)
        or torch.cuda.get_device_capability(left.device)[0] < 8
    ):
        raise ValueError(
            "Gram readout l=0 requires contiguous CUDA FP32 left/right="
            "(N,48,192) and gram=(48,48) tensors on compute capability 8.0+"
        )
    if out is not None and (
        tuple(out.shape) != (left.shape[0], HIDDEN_CHANNELS)
        or out.device != left.device
        or out.dtype != left.dtype
        or not out.is_contiguous()
        or out.data_ptr() % 16 != 0
    ):
        raise ValueError("readout l=0 output must be contiguous with shape (N,192)")


def run_readout_l0_gram(left, right, gram):
    """Run ``left[:, :, h]^T G right[:, :, h]`` in strict FP32."""
    import torch

    _validate_gram_tensors(left, right, gram)
    out = torch.empty(
        (left.shape[0], HIDDEN_CHANNELS),
        dtype=left.dtype,
        device=left.device,
    )
    _compiled_readout_l0_gram_forward(*_compile_key(left))(
        left,
        right,
        gram,
        out,
    )
    return out


def run_readout_l0(left, right, to_grid, from_grid):
    """Run the strict-FP32 row-zero readout forward."""
    _validate_tensors(left, right, to_grid, from_grid)
    from ..readout_l0 import (
        build_readout_l0_gram,
    )

    return run_readout_l0_gram(
        left,
        right,
        build_readout_l0_gram(to_grid, from_grid),
    )


def run_readout_l0_backward(dq0, left, right, to_grid, from_grid):
    """Run first backward and return full `(N,48,192)` input adjoints."""
    import torch

    _validate_tensors(left, right, to_grid, from_grid)
    if (
        tuple(dq0.shape) != (left.shape[0], HIDDEN_CHANNELS)
        or dq0.device != left.device
        or dq0.dtype != torch.float32
        or not dq0.is_contiguous()
    ):
        raise ValueError("dq0 must be contiguous CUDA FP32 with shape (N,192)")
    grad_left = torch.empty_like(left)
    grad_right = torch.empty_like(right)
    compile_key = _compile_key(left)
    from ..readout_l0 import (
        build_readout_l0_gram,
    )

    gram = build_readout_l0_gram(to_grid, from_grid)
    _compiled_readout_l0_gram_backward(*compile_key)(
        dq0,
        left,
        right,
        gram,
        grad_left,
        grad_right,
    )
    return grad_left, grad_right


__all__ = [
    "TiledReadoutL0GramBackward",
    "TiledReadoutL0GramForward",
    "run_readout_l0",
    "run_readout_l0_backward",
    "run_readout_l0_gram",
]
