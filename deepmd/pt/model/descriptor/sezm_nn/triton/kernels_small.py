# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN201
"""Specialized small-family Triton kernels for SeZM SO(2) rotations.

These kernels are the intended fast path for ``lmax <= 3``. They keep one
masked ``16x16`` Wigner tile in registers, so ``lmax=0`` and ``lmax=1`` can
share the same specialized family without paying the loop overhead of the
generic tiled kernels.
"""

from __future__ import (
    annotations,
)

import triton
import triton.language as tl

from .constants import TRITON_SMALL_FULL_DIM as TRITON_SMALL_FULL_DIM_VALUE

# Small kernels always materialize one padded ``16x16`` block and mask tails.
TRITON_SMALL_FULL_DIM = tl.constexpr(TRITON_SMALL_FULL_DIM_VALUE)


@triton.jit
def _load_full_wigner_matrix(
    wigner_ptr,
    edge_id,
    full_dim,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
) -> tl.tensor:
    """Load one padded ``16x16`` Wigner block in l-major order."""
    full_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
    full_mask = full_offsets < full_dim
    wigner_ptrs = (
        wigner_ptr
        + edge_id * wigner_stride_e
        + full_offsets[:, None] * wigner_stride_r
        + full_offsets[None, :] * wigner_stride_k
    )
    return tl.load(
        wigner_ptrs,
        mask=full_mask[:, None] & full_mask[None, :],
        other=0.0,
    )


@triton.jit
def _load_full_node_values(
    x_ptr,
    node_idx,
    full_dim,
    channel_offsets,
    channel_mask,
    x_stride_n,
    x_stride_d,
    x_stride_c,
) -> tl.tensor:
    """Load one padded ``16xC`` node feature block in l-major order."""
    full_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
    full_mask = full_offsets < full_dim
    x_ptrs = (
        x_ptr
        + node_idx * x_stride_n
        + full_offsets[:, None] * x_stride_d
        + channel_offsets[None, :] * x_stride_c
    )
    return tl.load(
        x_ptrs,
        mask=full_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )


@triton.jit
def _load_reduced_values_with_index(
    x_ptr,
    coeff_index_ptr,
    edge_id,
    reduced_dim,
    channel_offsets,
    channel_mask,
    x_stride_e,
    x_stride_r,
    x_stride_c,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    """Load reduced values together with the padded reduced->full row mapping."""
    reduced_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
    reduced_mask = reduced_offsets < reduced_dim
    x_ptrs = (
        x_ptr
        + edge_id * x_stride_e
        + reduced_offsets[:, None] * x_stride_r
        + channel_offsets[None, :] * x_stride_c
    )
    reduced_values = tl.load(
        x_ptrs,
        mask=reduced_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    coeff_rows = tl.load(
        coeff_index_ptr + reduced_offsets,
        mask=reduced_mask,
        other=-1,
    ).to(tl.int64)
    return reduced_values, reduced_mask, coeff_rows


@triton.jit
def _scatter_reduced_to_full_matrix(
    reduced_values,
    reduced_mask,
    coeff_rows,
    BLOCK_CHANNEL: tl.constexpr,
) -> tl.tensor:
    """Scatter a padded reduced block into a padded full l-major block."""
    row_ids = tl.arange(0, TRITON_SMALL_FULL_DIM)
    full_values = tl.zeros(
        (TRITON_SMALL_FULL_DIM, BLOCK_CHANNEL),
        dtype=reduced_values.dtype,
    )
    for row in range(TRITON_SMALL_FULL_DIM):
        row_mask = (coeff_rows == row)[:, None] & reduced_mask[:, None]
        row_value = tl.sum(tl.where(row_mask, reduced_values, 0.0), axis=0).to(
            reduced_values.dtype
        )
        full_values = tl.where(
            row_ids[:, None] == row,
            row_value[None, :],
            full_values,
        )
    return full_values


@triton.jit
def _select_reduced_from_full_matrix(
    full_values,
    reduced_mask,
    coeff_rows,
    BLOCK_CHANNEL: tl.constexpr,
) -> tl.tensor:
    """Select reduced rows from a padded full l-major block."""
    row_ids = tl.arange(0, TRITON_SMALL_FULL_DIM)
    reduced_values = tl.zeros(
        (TRITON_SMALL_FULL_DIM, BLOCK_CHANNEL),
        dtype=full_values.dtype,
    )
    for row in range(TRITON_SMALL_FULL_DIM):
        row_value = tl.sum(
            tl.where(row_ids[:, None] == row, full_values, 0.0),
            axis=0,
        ).to(full_values.dtype)
        reduced_values = tl.where(
            (coeff_rows == row)[:, None] & reduced_mask[:, None],
            row_value[None, :],
            reduced_values,
        )
    return reduced_values


@triton.jit
def _build_full_matrix_l1(
    y0,
    y1,
    y2,
    y3,
    BLOCK_CHANNEL: tl.constexpr,
) -> tl.tensor:
    """Build a padded full matrix from the ``lmax=1`` row vectors."""
    row_ids = tl.arange(0, TRITON_SMALL_FULL_DIM)
    full_values = tl.zeros(
        (TRITON_SMALL_FULL_DIM, BLOCK_CHANNEL),
        dtype=tl.float32,
    )
    full_values = tl.where(row_ids[:, None] == 0, y0[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 1, y1[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 2, y2[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 3, y3[None, :], full_values)
    return full_values


@triton.jit
def _build_full_matrix_l2(
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    y8,
    BLOCK_CHANNEL: tl.constexpr,
) -> tl.tensor:
    """Build a padded full matrix from the ``lmax=2`` row vectors."""
    row_ids = tl.arange(0, TRITON_SMALL_FULL_DIM)
    full_values = tl.zeros(
        (TRITON_SMALL_FULL_DIM, BLOCK_CHANNEL),
        dtype=tl.float32,
    )
    full_values = tl.where(row_ids[:, None] == 0, y0[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 1, y1[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 2, y2[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 3, y3[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 4, y4[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 5, y5[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 6, y6[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 7, y7[None, :], full_values)
    full_values = tl.where(row_ids[:, None] == 8, y8[None, :], full_values)
    return full_values


@triton.jit
def _matvec_l1(
    w_full,
    x_full,
) -> tl.tensor:
    """Apply the packed ``lmax=1`` block-diagonal Wigner matrix."""
    return tl.dot(w_full.to(x_full.dtype), x_full, input_precision="ieee")


@triton.jit
def _matvec_t_l1(
    w_full,
    x_full,
) -> tl.tensor:
    """Apply the transpose of the packed ``lmax=1`` Wigner matrix."""
    return tl.dot(
        tl.trans(w_full.to(x_full.dtype)),
        x_full,
        input_precision="ieee",
    )


@triton.jit
def _matvec_l2(
    w_full,
    x_full,
) -> tl.tensor:
    """Apply the packed ``lmax=2`` block-diagonal Wigner matrix."""
    return tl.dot(w_full.to(x_full.dtype), x_full, input_precision="ieee")


@triton.jit
def _matvec_t_l2(
    w_full,
    x_full,
) -> tl.tensor:
    """Apply the transpose of the packed ``lmax=2`` Wigner matrix."""
    return tl.dot(
        tl.trans(w_full.to(x_full.dtype)),
        x_full,
        input_precision="ieee",
    )


@triton.jit
def rotate_to_local_l1_forward_kernel(
    x_ptr,
    src_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Fused ``global -> local reduced`` rotation specialized for ``lmax=1``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        src_idx = tl.load(src_ptr + edge_id).to(tl.int64)
        coeff_rows = tl.load(
            coeff_index_ptr + tl.arange(0, TRITON_SMALL_FULL_DIM),
            mask=tl.arange(0, TRITON_SMALL_FULL_DIM) < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < reduced_dim
        x_full = _load_full_node_values(
            x_ptr,
            src_idx,
            full_dim,
            channel_offsets,
            channel_mask,
            x_stride_n,
            x_stride_d,
            x_stride_c,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(x_full.dtype)
        y_full = _matvec_l1(w_full, x_full)
        out_values = _select_reduced_from_full_matrix(
            y_full,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * out_stride_r
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            out_values,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_l2_forward_kernel(
    x_ptr,
    src_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Fused ``global -> local reduced`` rotation specialized for ``lmax=2``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        src_idx = tl.load(src_ptr + edge_id).to(tl.int64)
        coeff_rows = tl.load(
            coeff_index_ptr + tl.arange(0, TRITON_SMALL_FULL_DIM),
            mask=tl.arange(0, TRITON_SMALL_FULL_DIM) < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < reduced_dim
        x_full = _load_full_node_values(
            x_ptr,
            src_idx,
            full_dim,
            channel_offsets,
            channel_mask,
            x_stride_n,
            x_stride_d,
            x_stride_c,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(x_full.dtype)
        y_full = _matvec_l2(w_full, x_full)
        out_values = _select_reduced_from_full_matrix(
            y_full,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * out_stride_r
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            out_values,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_l3_forward_kernel(
    x_ptr,
    src_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Fused ``global -> local reduced`` rotation specialized for ``lmax=3``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        src_idx = tl.load(src_ptr + edge_id).to(tl.int64)
        coeff_rows = tl.load(
            coeff_index_ptr + tl.arange(0, TRITON_SMALL_FULL_DIM),
            mask=tl.arange(0, TRITON_SMALL_FULL_DIM) < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < reduced_dim
        x_full = _load_full_node_values(
            x_ptr,
            src_idx,
            full_dim,
            channel_offsets,
            channel_mask,
            x_stride_n,
            x_stride_d,
            x_stride_c,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(x_full.dtype)
        y_full = tl.dot(w_full, x_full, input_precision="ieee")
        out_values = _select_reduced_from_full_matrix(
            y_full,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * out_stride_r
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            out_values,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_l1_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_edge_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute per-edge source gradients specialized for ``lmax=1``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        grad_reduced, reduced_mask, coeff_rows = _load_reduced_values_with_index(
            grad_out_ptr,
            coeff_index_ptr,
            edge_id,
            reduced_dim,
            channel_offsets,
            channel_mask,
            grad_out_stride_e,
            grad_out_stride_r,
            grad_out_stride_c,
        )
        grad_full = _scatter_reduced_to_full_matrix(
            grad_reduced,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(grad_full.dtype)
        dx_full = _matvec_t_l1(w_full, grad_full)
        full_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < full_dim
        grad_edge_ptrs = (
            grad_edge_ptr
            + edge_id * grad_edge_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * grad_edge_stride_d
            + channel_offsets[None, :] * grad_edge_stride_c
        )
        tl.store(
            grad_edge_ptrs,
            dx_full,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_l2_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_edge_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute per-edge source gradients specialized for ``lmax=2``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        grad_reduced, reduced_mask, coeff_rows = _load_reduced_values_with_index(
            grad_out_ptr,
            coeff_index_ptr,
            edge_id,
            reduced_dim,
            channel_offsets,
            channel_mask,
            grad_out_stride_e,
            grad_out_stride_r,
            grad_out_stride_c,
        )
        grad_full = _scatter_reduced_to_full_matrix(
            grad_reduced,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(grad_full.dtype)
        dx_full = _matvec_t_l2(w_full, grad_full)
        full_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < full_dim
        grad_edge_ptrs = (
            grad_edge_ptr
            + edge_id * grad_edge_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * grad_edge_stride_d
            + channel_offsets[None, :] * grad_edge_stride_c
        )
        tl.store(
            grad_edge_ptrs,
            dx_full,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_l3_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_edge_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute per-edge source gradients specialized for ``lmax=3``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        grad_reduced, reduced_mask, coeff_rows = _load_reduced_values_with_index(
            grad_out_ptr,
            coeff_index_ptr,
            edge_id,
            reduced_dim,
            channel_offsets,
            channel_mask,
            grad_out_stride_e,
            grad_out_stride_r,
            grad_out_stride_c,
        )
        grad_full = _scatter_reduced_to_full_matrix(
            grad_reduced,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(grad_full.dtype)
        dx_full = tl.dot(
            tl.trans(w_full),
            grad_full,
            input_precision="ieee",
        )
        full_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < full_dim
        grad_edge_ptrs = (
            grad_edge_ptr
            + edge_id * grad_edge_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * grad_edge_stride_d
            + channel_offsets[None, :] * grad_edge_stride_c
        )
        tl.store(
            grad_edge_ptrs,
            dx_full,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_to_local_small_bwd_dw_kernel(
    grad_out_ptr,
    x_ptr,
    src_ptr,
    coeff_index_ptr,
    grad_wigner_ptr,
    num_edges,
    reduced_dim,
    full_dim,
    channels,
    grad_out_stride_e,
    grad_out_stride_r,
    grad_out_stride_c,
    x_stride_n,
    x_stride_d,
    x_stride_c,
    grad_wigner_stride_e,
    grad_wigner_stride_r,
    grad_wigner_stride_k,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute full Wigner gradients for specialized small-l rotate-to-local."""
    edge_id = tl.program_id(0)
    channel_offsets = tl.arange(0, BLOCK_CHANNEL)
    full_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
    while edge_id < num_edges:
        coeff_rows = tl.load(
            coeff_index_ptr + full_offsets,
            mask=full_offsets < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = full_offsets < reduced_dim
        src_idx = tl.load(src_ptr + edge_id).to(tl.int64)
        grad_w_acc = tl.zeros(
            (TRITON_SMALL_FULL_DIM, TRITON_SMALL_FULL_DIM),
            dtype=tl.float32,
        )
        channel_start = 0
        while channel_start < channels:
            block_offsets = channel_start + channel_offsets
            channel_mask = block_offsets < channels
            grad_reduced, _, _ = _load_reduced_values_with_index(
                grad_out_ptr,
                coeff_index_ptr,
                edge_id,
                reduced_dim,
                block_offsets,
                channel_mask,
                grad_out_stride_e,
                grad_out_stride_r,
                grad_out_stride_c,
            )
            grad_full_block = _scatter_reduced_to_full_matrix(
                grad_reduced,
                reduced_mask,
                coeff_rows,
                BLOCK_CHANNEL=BLOCK_CHANNEL,
            )
            x_full_block = _load_full_node_values(
                x_ptr,
                src_idx,
                full_dim,
                block_offsets,
                channel_mask,
                x_stride_n,
                x_stride_d,
                x_stride_c,
            )
            grad_w_acc += tl.dot(
                grad_full_block,
                tl.trans(x_full_block.to(grad_full_block.dtype)),
                input_precision="ieee",
            )
            channel_start += BLOCK_CHANNEL
        grad_w_ptrs = (
            grad_wigner_ptr
            + edge_id * grad_wigner_stride_e
            + full_offsets[:, None] * grad_wigner_stride_r
            + full_offsets[None, :] * grad_wigner_stride_k
        )
        full_mask = full_offsets < full_dim
        tl.store(
            grad_w_ptrs,
            grad_w_acc,
            mask=full_mask[:, None] & full_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_l1_forward_kernel(
    x_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    full_dim,
    channels,
    x_stride_e,
    x_stride_r,
    x_stride_c,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
    out_stride_e,
    out_stride_d,
    out_stride_c,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Fused ``local reduced -> global`` rotation specialized for ``lmax=1``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        reduced_values, reduced_mask, coeff_rows = _load_reduced_values_with_index(
            x_ptr,
            coeff_index_ptr,
            edge_id,
            reduced_dim,
            channel_offsets,
            channel_mask,
            x_stride_e,
            x_stride_r,
            x_stride_c,
        )
        x_full = _scatter_reduced_to_full_matrix(
            reduced_values,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(x_full.dtype)
        y_full = _matvec_l1(w_full, x_full)
        full_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < full_dim
        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * out_stride_d
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            y_full,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_l2_forward_kernel(
    x_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    full_dim,
    channels,
    x_stride_e,
    x_stride_r,
    x_stride_c,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
    out_stride_e,
    out_stride_d,
    out_stride_c,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Fused ``local reduced -> global`` rotation specialized for ``lmax=2``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        reduced_values, reduced_mask, coeff_rows = _load_reduced_values_with_index(
            x_ptr,
            coeff_index_ptr,
            edge_id,
            reduced_dim,
            channel_offsets,
            channel_mask,
            x_stride_e,
            x_stride_r,
            x_stride_c,
        )
        x_full = _scatter_reduced_to_full_matrix(
            reduced_values,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(x_full.dtype)
        y_full = _matvec_l2(w_full, x_full)
        full_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < full_dim
        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * out_stride_d
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            y_full,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_l3_forward_kernel(
    x_ptr,
    wigner_ptr,
    coeff_index_ptr,
    out_ptr,
    num_edges,
    reduced_dim,
    full_dim,
    channels,
    x_stride_e,
    x_stride_r,
    x_stride_c,
    wigner_stride_e,
    wigner_stride_r,
    wigner_stride_k,
    out_stride_e,
    out_stride_d,
    out_stride_c,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Fused ``local reduced -> global`` rotation specialized for ``lmax=3``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        reduced_values, reduced_mask, coeff_rows = _load_reduced_values_with_index(
            x_ptr,
            coeff_index_ptr,
            edge_id,
            reduced_dim,
            channel_offsets,
            channel_mask,
            x_stride_e,
            x_stride_r,
            x_stride_c,
        )
        x_full = _scatter_reduced_to_full_matrix(
            reduced_values,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(x_full.dtype)
        y_full = tl.dot(
            w_full.to(x_full.dtype),
            x_full,
            input_precision="ieee",
        )
        full_mask = tl.arange(0, TRITON_SMALL_FULL_DIM) < full_dim
        out_ptrs = (
            out_ptr
            + edge_id * out_stride_e
            + tl.arange(0, TRITON_SMALL_FULL_DIM)[:, None] * out_stride_d
            + channel_offsets[None, :] * out_stride_c
        )
        tl.store(
            out_ptrs,
            y_full,
            mask=full_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_l1_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_x_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute reduced-layout gradients specialized for ``lmax=1``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        full_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
        full_mask = full_offsets < full_dim
        grad_ptrs = (
            grad_out_ptr
            + edge_id * grad_out_stride_e
            + full_offsets[:, None] * grad_out_stride_d
            + channel_offsets[None, :] * grad_out_stride_c
        )
        grad_full = tl.load(
            grad_ptrs,
            mask=full_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        coeff_rows = tl.load(
            coeff_index_ptr + full_offsets,
            mask=full_offsets < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = full_offsets < reduced_dim
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(grad_full.dtype)
        dx_full = _matvec_t_l1(w_full, grad_full)
        grad_reduced = _select_reduced_from_full_matrix(
            dx_full,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        grad_x_ptrs = (
            grad_x_ptr
            + edge_id * grad_x_stride_e
            + full_offsets[:, None] * grad_x_stride_r
            + channel_offsets[None, :] * grad_x_stride_c
        )
        tl.store(
            grad_x_ptrs,
            grad_reduced,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_l2_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_x_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute reduced-layout gradients specialized for ``lmax=2``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        full_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
        full_mask = full_offsets < full_dim
        grad_ptrs = (
            grad_out_ptr
            + edge_id * grad_out_stride_e
            + full_offsets[:, None] * grad_out_stride_d
            + channel_offsets[None, :] * grad_out_stride_c
        )
        grad_full = tl.load(
            grad_ptrs,
            mask=full_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        coeff_rows = tl.load(
            coeff_index_ptr + full_offsets,
            mask=full_offsets < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = full_offsets < reduced_dim
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(grad_full.dtype)
        dx_full = _matvec_t_l2(w_full, grad_full)
        grad_reduced = _select_reduced_from_full_matrix(
            dx_full,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        grad_x_ptrs = (
            grad_x_ptr
            + edge_id * grad_x_stride_e
            + full_offsets[:, None] * grad_x_stride_r
            + channel_offsets[None, :] * grad_x_stride_c
        )
        tl.store(
            grad_x_ptrs,
            grad_reduced,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_l3_bwd_dx_kernel(
    grad_out_ptr,
    wigner_ptr,
    coeff_index_ptr,
    grad_x_ptr,
    num_edges,
    reduced_dim,
    full_dim,
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
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute reduced-layout gradients specialized for ``lmax=3``."""
    edge_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)
    channel_offsets = channel_block_id * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    channel_mask = channel_offsets < channels
    while edge_id < num_edges:
        full_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
        full_mask = full_offsets < full_dim
        grad_ptrs = (
            grad_out_ptr
            + edge_id * grad_out_stride_e
            + full_offsets[:, None] * grad_out_stride_d
            + channel_offsets[None, :] * grad_out_stride_c
        )
        grad_full = tl.load(
            grad_ptrs,
            mask=full_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        coeff_rows = tl.load(
            coeff_index_ptr + full_offsets,
            mask=full_offsets < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = full_offsets < reduced_dim
        w_full = _load_full_wigner_matrix(
            wigner_ptr,
            edge_id,
            full_dim,
            wigner_stride_e,
            wigner_stride_r,
            wigner_stride_k,
        ).to(grad_full.dtype)
        dx_full = tl.dot(
            tl.trans(w_full.to(grad_full.dtype)),
            grad_full,
            input_precision="ieee",
        )
        grad_reduced = _select_reduced_from_full_matrix(
            dx_full,
            reduced_mask,
            coeff_rows,
            BLOCK_CHANNEL=BLOCK_CHANNEL,
        )
        grad_x_ptrs = (
            grad_x_ptr
            + edge_id * grad_x_stride_e
            + full_offsets[:, None] * grad_x_stride_r
            + channel_offsets[None, :] * grad_x_stride_c
        )
        tl.store(
            grad_x_ptrs,
            grad_reduced,
            mask=reduced_mask[:, None] & channel_mask[None, :],
        )
        edge_id += GRID_E_STRIDE


@triton.jit
def rotate_back_small_bwd_dw_kernel(
    grad_out_ptr,
    x_ptr,
    coeff_index_ptr,
    grad_wigner_ptr,
    num_edges,
    reduced_dim,
    full_dim,
    channels,
    grad_out_stride_e,
    grad_out_stride_d,
    grad_out_stride_c,
    x_stride_e,
    x_stride_r,
    x_stride_c,
    grad_wigner_stride_e,
    grad_wigner_stride_r,
    grad_wigner_stride_k,
    BLOCK_CHANNEL: tl.constexpr,
    GRID_E_STRIDE: tl.constexpr,
):
    """Compute full Wigner gradients for specialized small-l rotate-back."""
    edge_id = tl.program_id(0)
    channel_offsets = tl.arange(0, BLOCK_CHANNEL)
    full_offsets = tl.arange(0, TRITON_SMALL_FULL_DIM)
    while edge_id < num_edges:
        coeff_rows = tl.load(
            coeff_index_ptr + full_offsets,
            mask=full_offsets < reduced_dim,
            other=-1,
        ).to(tl.int64)
        reduced_mask = full_offsets < reduced_dim
        grad_w_acc = tl.zeros(
            (TRITON_SMALL_FULL_DIM, TRITON_SMALL_FULL_DIM),
            dtype=tl.float32,
        )
        channel_start = 0
        while channel_start < channels:
            block_offsets = channel_start + channel_offsets
            channel_mask = block_offsets < channels
            full_mask = full_offsets < full_dim
            grad_ptrs = (
                grad_out_ptr
                + edge_id * grad_out_stride_e
                + full_offsets[:, None] * grad_out_stride_d
                + block_offsets[None, :] * grad_out_stride_c
            )
            grad_full = tl.load(
                grad_ptrs,
                mask=full_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            reduced_values, _, _ = _load_reduced_values_with_index(
                x_ptr,
                coeff_index_ptr,
                edge_id,
                reduced_dim,
                block_offsets,
                channel_mask,
                x_stride_e,
                x_stride_r,
                x_stride_c,
            )
            x_full = _scatter_reduced_to_full_matrix(
                reduced_values,
                reduced_mask,
                coeff_rows,
                BLOCK_CHANNEL=BLOCK_CHANNEL,
            )
            grad_w_acc += tl.dot(
                grad_full,
                tl.trans(x_full.to(grad_full.dtype)),
                input_precision="ieee",
            )
            channel_start += BLOCK_CHANNEL
        grad_w_ptrs = (
            grad_wigner_ptr
            + edge_id * grad_wigner_stride_e
            + full_offsets[:, None] * grad_wigner_stride_r
            + full_offsets[None, :] * grad_wigner_stride_k
        )
        full_mask = full_offsets < full_dim
        tl.store(
            grad_w_ptrs,
            grad_w_acc,
            mask=full_mask[:, None] & full_mask[None, :],
        )
        edge_id += GRID_E_STRIDE
