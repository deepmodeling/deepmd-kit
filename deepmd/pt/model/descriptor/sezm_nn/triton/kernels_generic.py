# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN201
"""Generic tiled Triton kernels for SeZM SO(2) rotation hot paths.

This file holds the variable-``lmax`` family used once the packed SO(3) block
no longer fits the small specialized kernels. The tile sizes are fixed on
purpose: ``BLOCK_FULL == BLOCK_REDUCED == 16`` keeps every ``tl.dot`` on a CUDA
shape that Triton accepts, and the kernels below explicitly request
``input_precision="ieee"`` so float32 matches eager PyTorch instead of TF32.
"""

from __future__ import (
    annotations,
)

import triton
import triton.language as tl

# Keep both contraction dimensions at 16 so Triton always sees a legal dot tile.


@triton.jit
def rotate_to_local_forward_kernel(
    x_ptr,
    src_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    dim_full,
    channels,
    x_stride_n,
    x_stride_d,
    x_stride_c,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
    out_stride_e,
    out_stride_r,
    out_stride_c,
    BLOCK_REDUCED: tl.constexpr,
    BLOCK_FULL: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute fused row-projected Wigner rotation ``D_to_m @ x[src]``."""
    edge_id = tl.program_id(0)
    reduced_block_id = tl.program_id(1)
    channel_block_id = tl.program_id(2)

    reduced_offsets = reduced_block_id * BLOCK_REDUCED + tl.arange(0, BLOCK_REDUCED)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    reduced_mask = reduced_offsets < reduced_dim
    channel_mask = channel_offsets < channels

    while edge_id < num_edges:
        src_idx = tl.load(src_ptr + edge_id).to(tl.int64)
        coeff_rows = tl.load(
            coeff_index_ptr + reduced_offsets,
            mask=reduced_mask,
            other=0,
        ).to(tl.int64)
        acc = tl.zeros((BLOCK_REDUCED, BLOCK_CHANNEL), dtype=tl.float32)

        for full_block in range(0, tl.cdiv(dim_full, BLOCK_FULL)):
            full_offsets = full_block * BLOCK_FULL + tl.arange(0, BLOCK_FULL)
            full_mask = full_offsets < dim_full
            wigner_ptrs = (
                wigner_ptr
                + edge_id * wigner_stride_e
                + coeff_rows[:, None] * wigner_stride_r
                + full_offsets[None, :] * wigner_stride_k
            )
            x_ptrs = (
                x_ptr
                + src_idx * x_stride_n
                + full_offsets[:, None] * x_stride_d
                + channel_offsets[None, :] * x_stride_c
            )
            w_block = tl.load(
                wigner_ptrs,
                mask=reduced_mask[:, None] & full_mask[None, :],
                other=0.0,
            )
            x_block = tl.load(
                x_ptrs,
                mask=full_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            # Match the eager autocast path: rotate in the activation dtype chosen
            # by the current AMP context instead of forcing a higher Wigner dtype.
            w_block = w_block.to(x_block.dtype)
            acc = tl.dot(
                w_block,
                x_block,
                acc,
                input_precision="ieee",
            )

        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + reduced_offsets[:, None] * out_stride_r
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            acc,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_edge_ptr,
    num_edges,
    reduced_dim,
    dim_full,
    channels,
    grad_out_stride_e,
    grad_out_stride_r,
    grad_out_stride_c,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
    grad_edge_stride_e,
    grad_edge_stride_d,
    grad_edge_stride_c,
    BLOCK_REDUCED: tl.constexpr,
    BLOCK_FULL: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute per-edge source gradients ``D_to_m^T @ grad`` before scatter."""
    edge_id = tl.program_id(0)
    full_block_id = tl.program_id(1)
    channel_block_id = tl.program_id(2)

    full_offsets = full_block_id * BLOCK_FULL + tl.arange(0, BLOCK_FULL)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    full_mask = full_offsets < dim_full
    channel_mask = channel_offsets < channels

    while edge_id < num_edges:
        acc = tl.zeros((BLOCK_FULL, BLOCK_CHANNEL), dtype=tl.float32)

        for reduced_block in range(0, tl.cdiv(reduced_dim, BLOCK_REDUCED)):
            reduced_offsets = reduced_block * BLOCK_REDUCED + tl.arange(
                0, BLOCK_REDUCED
            )
            reduced_mask = reduced_offsets < reduced_dim
            coeff_rows = tl.load(
                coeff_index_ptr + reduced_offsets,
                mask=reduced_mask,
                other=0,
            ).to(tl.int64)
            wigner_ptrs = (
                wigner_ptr
                + edge_id * wigner_stride_e
                + coeff_rows[:, None] * wigner_stride_r
                + full_offsets[None, :] * wigner_stride_k
            )
            grad_ptrs = (
                grad_out_ptr
                + edge_id * grad_out_stride_e
                + reduced_offsets[:, None] * grad_out_stride_r
                + channel_offsets[None, :] * grad_out_stride_c
            )
            w_block = tl.load(
                wigner_ptrs,
                mask=reduced_mask[:, None] & full_mask[None, :],
                other=0.0,
            )
            grad_block = tl.load(
                grad_ptrs,
                mask=reduced_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            w_block = w_block.to(grad_block.dtype)
            acc = tl.dot(
                tl.trans(w_block),
                grad_block,
                acc,
                input_precision="ieee",
            )

        grad_edge_ptrs = (
            grad_edge_ptr
            + edge_id * grad_edge_stride_e
            + full_offsets[:, None] * grad_edge_stride_d
            + channel_offsets[None, :] * grad_edge_stride_c
        )
        tl.store(
            grad_edge_ptrs,
            acc,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_bwd_dw_kernel(
    grad_out_ptr,
    x_ptr,
    src_ptr,
    coeff_index_ptr,
    grad_rows_ptr,
    num_edges,
    reduced_dim,
    dim_full,
    channels,
    grad_out_stride_e,
    grad_out_stride_r,
    grad_out_stride_c,
    x_stride_n,
    x_stride_d,
    x_stride_c,
    grad_rows_stride_e,
    grad_rows_stride_r,
    grad_rows_stride_d,
    BLOCK_REDUCED: tl.constexpr,
    BLOCK_FULL: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute row-selected Wigner gradients ``grad @ x[src]^T``."""
    edge_id = tl.program_id(0)
    reduced_block_id = tl.program_id(1)
    full_block_id = tl.program_id(2)

    reduced_offsets = reduced_block_id * BLOCK_REDUCED + tl.arange(0, BLOCK_REDUCED)
    full_offsets = full_block_id * BLOCK_FULL + tl.arange(0, BLOCK_FULL)
    reduced_mask = reduced_offsets < reduced_dim
    full_mask = full_offsets < dim_full

    while edge_id < num_edges:
        src_idx = tl.load(src_ptr + edge_id).to(tl.int64)
        acc = tl.zeros((BLOCK_REDUCED, BLOCK_FULL), dtype=tl.float32)

        for channel_block in range(0, tl.cdiv(channels, BLOCK_CHANNEL)):
            channel_offsets = channel_block * BLOCK_CHANNEL + tl.arange(
                0, BLOCK_CHANNEL
            )
            channel_mask = channel_offsets < channels
            grad_ptrs = (
                grad_out_ptr
                + edge_id * grad_out_stride_e
                + reduced_offsets[:, None] * grad_out_stride_r
                + channel_offsets[None, :] * grad_out_stride_c
            )
            x_ptrs = (
                x_ptr
                + src_idx * x_stride_n
                + full_offsets[:, None] * x_stride_d
                + channel_offsets[None, :] * x_stride_c
            )
            grad_block = tl.load(
                grad_ptrs,
                mask=reduced_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            x_block = tl.load(
                x_ptrs,
                mask=full_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(
                grad_block,
                tl.trans(x_block),
                acc,
                input_precision="ieee",
            )

        grad_rows_ptrs = (
            grad_rows_ptr
            + edge_id * grad_rows_stride_e
            + reduced_offsets[:, None] * grad_rows_stride_r
            + full_offsets[None, :] * grad_rows_stride_d
        )
        tl.store(
            grad_rows_ptrs,
            acc,
            mask=reduced_mask[:, None] & full_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_forward_kernel(
    x_local_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    dim_full,
    channels,
    x_local_stride_e,
    x_local_stride_r,
    x_local_stride_c,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
    out_stride_e,
    out_stride_d,
    out_stride_c,
    BLOCK_REDUCED: tl.constexpr,
    BLOCK_FULL: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute fused inverse rotation ``Dt_from_m @ x_local``."""
    edge_id = tl.program_id(0)
    full_block_id = tl.program_id(1)
    channel_block_id = tl.program_id(2)

    full_offsets = full_block_id * BLOCK_FULL + tl.arange(0, BLOCK_FULL)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    full_mask = full_offsets < dim_full
    channel_mask = channel_offsets < channels

    while edge_id < num_edges:
        acc = tl.zeros((BLOCK_FULL, BLOCK_CHANNEL), dtype=tl.float32)

        for reduced_block in range(0, tl.cdiv(reduced_dim, BLOCK_REDUCED)):
            reduced_offsets = reduced_block * BLOCK_REDUCED + tl.arange(
                0, BLOCK_REDUCED
            )
            reduced_mask = reduced_offsets < reduced_dim
            coeff_cols = tl.load(
                coeff_index_ptr + reduced_offsets,
                mask=reduced_mask,
                other=0,
            ).to(tl.int64)
            wigner_ptrs = (
                wigner_ptr
                + edge_id * wigner_stride_e
                + full_offsets[:, None] * wigner_stride_r
                + coeff_cols[None, :] * wigner_stride_k
            )
            x_ptrs = (
                x_local_ptr
                + edge_id * x_local_stride_e
                + reduced_offsets[:, None] * x_local_stride_r
                + channel_offsets[None, :] * x_local_stride_c
            )
            w_block = tl.load(
                wigner_ptrs,
                mask=full_mask[:, None] & reduced_mask[None, :],
                other=0.0,
            )
            x_block = tl.load(
                x_ptrs,
                mask=reduced_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            w_block = w_block.to(x_block.dtype)
            acc = tl.dot(
                w_block,
                x_block,
                acc,
                input_precision="ieee",
            )

        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + full_offsets[:, None] * out_stride_d
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            acc,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_x_ptr,
    num_edges,
    reduced_dim,
    dim_full,
    channels,
    grad_out_stride_e,
    grad_out_stride_d,
    grad_out_stride_c,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
    grad_x_stride_e,
    grad_x_stride_r,
    grad_x_stride_c,
    BLOCK_REDUCED: tl.constexpr,
    BLOCK_FULL: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute reduced-layout gradients ``Dt_from_m^T @ grad``."""
    edge_id = tl.program_id(0)
    reduced_block_id = tl.program_id(1)
    channel_block_id = tl.program_id(2)

    reduced_offsets = reduced_block_id * BLOCK_REDUCED + tl.arange(0, BLOCK_REDUCED)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    reduced_mask = reduced_offsets < reduced_dim
    channel_mask = channel_offsets < channels

    while edge_id < num_edges:
        coeff_cols = tl.load(
            coeff_index_ptr + reduced_offsets,
            mask=reduced_mask,
            other=0,
        ).to(tl.int64)
        acc = tl.zeros((BLOCK_REDUCED, BLOCK_CHANNEL), dtype=tl.float32)

        for full_block in range(0, tl.cdiv(dim_full, BLOCK_FULL)):
            full_offsets = full_block * BLOCK_FULL + tl.arange(0, BLOCK_FULL)
            full_mask = full_offsets < dim_full
            wigner_ptrs = (
                wigner_ptr
                + edge_id * wigner_stride_e
                + full_offsets[:, None] * wigner_stride_r
                + coeff_cols[None, :] * wigner_stride_k
            )
            grad_ptrs = (
                grad_out_ptr
                + edge_id * grad_out_stride_e
                + full_offsets[:, None] * grad_out_stride_d
                + channel_offsets[None, :] * grad_out_stride_c
            )
            w_block = tl.load(
                wigner_ptrs,
                mask=full_mask[:, None] & reduced_mask[None, :],
                other=0.0,
            )
            grad_block = tl.load(
                grad_ptrs,
                mask=full_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            w_block = w_block.to(grad_block.dtype)
            acc = tl.dot(
                tl.trans(w_block),
                grad_block,
                acc,
                input_precision="ieee",
            )

        grad_x_ptrs = (
            grad_x_ptr
            + edge_id * grad_x_stride_e
            + reduced_offsets[:, None] * grad_x_stride_r
            + channel_offsets[None, :] * grad_x_stride_c
        )
        tl.store(
            grad_x_ptrs,
            acc,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_bwd_dw_kernel(
    grad_out_ptr,
    x_local_ptr,
    grad_cols_ptr,
    num_edges,
    reduced_dim,
    dim_full,
    channels,
    grad_out_stride_e,
    grad_out_stride_d,
    grad_out_stride_c,
    x_local_stride_e,
    x_local_stride_r,
    x_local_stride_c,
    grad_cols_stride_e,
    grad_cols_stride_d,
    grad_cols_stride_r,
    BLOCK_REDUCED: tl.constexpr,
    BLOCK_FULL: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute column-selected inverse Wigner gradients ``grad @ x_local^T``."""
    edge_id = tl.program_id(0)
    full_block_id = tl.program_id(1)
    reduced_block_id = tl.program_id(2)

    full_offsets = full_block_id * BLOCK_FULL + tl.arange(0, BLOCK_FULL)
    reduced_offsets = reduced_block_id * BLOCK_REDUCED + tl.arange(0, BLOCK_REDUCED)
    full_mask = full_offsets < dim_full
    reduced_mask = reduced_offsets < reduced_dim

    while edge_id < num_edges:
        acc = tl.zeros((BLOCK_FULL, BLOCK_REDUCED), dtype=tl.float32)

        for channel_block in range(0, tl.cdiv(channels, BLOCK_CHANNEL)):
            channel_offsets = channel_block * BLOCK_CHANNEL + tl.arange(
                0, BLOCK_CHANNEL
            )
            channel_mask = channel_offsets < channels
            grad_ptrs = (
                grad_out_ptr
                + edge_id * grad_out_stride_e
                + full_offsets[:, None] * grad_out_stride_d
                + channel_offsets[None, :] * grad_out_stride_c
            )
            x_ptrs = (
                x_local_ptr
                + edge_id * x_local_stride_e
                + reduced_offsets[:, None] * x_local_stride_r
                + channel_offsets[None, :] * x_local_stride_c
            )
            grad_block = tl.load(
                grad_ptrs,
                mask=full_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            x_block = tl.load(
                x_ptrs,
                mask=reduced_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(
                grad_block,
                tl.trans(x_block),
                acc,
                input_precision="ieee",
            )

        grad_cols_ptrs = (
            grad_cols_ptr
            + edge_id * grad_cols_stride_e
            + full_offsets[:, None] * grad_cols_stride_d
            + reduced_offsets[None, :] * grad_cols_stride_r
        )
        tl.store(
            grad_cols_ptrs,
            acc,
            mask=full_mask[:, None] & reduced_mask[None, :],
        )
        edge_id += GRID_E_STRIDE
