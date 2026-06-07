# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Fused Triton SO(2)/Wigner rotation operators for the SeZM/DPA4 descriptor.

This module provides a *clean room* Triton implementation of the two rotation
hot paths used by the SeZM SO(2) convolution:

``rotate_to_local`` (global -> edge-local reduced frame)
    For every edge ``e`` with source node ``src[e]``::

        out[e] = Wrows[e] @ x[src[e]]  # (Dm, C)
        Wrows[e][m, k] = wigner[e, coeff_index[m], k]  # (Dm, D), k < D

    i.e. the eager reference ``bmm(D_to_m, x[src])`` where
    ``D_to_m = wigner[:, :D, :D].index_select(1, coeff_index)``.

``rotate_back`` (edge-local reduced frame -> global)
    For every edge ``e``::

        out[e] = Wcols[e] @ x_local[e]  # (D, C)
        Wcols[e][d, m] = wigner[e, d, coeff_index[m]]  # (D, Dm), d < D

    i.e. the eager reference ``bmm(Dt_from_m, x_local)`` where
    ``Dt_from_m = wigner[:, :D, :D].index_select(2, coeff_index)``.

Design goals
------------
1. **Fuse the gathers into the GEMM.** The eager / ``torch.compile`` path first
   materializes ``D_to_m`` (or ``Dt_from_m``), shape ``(E, Dm, D)``, *and*
   ``x[src]``, shape ``(E, D, C)``, before calling ``bmm``. For lmax 10 with
   E=100k that is ~9 GB of scratch that is written and immediately re-read.
   We instead gather the Wigner rows/columns (by ``coeff_index``) and the node
   features (by ``src``) *inside* the kernel, so neither scratch tensor is ever
   created. Each edge is one tiny GEMM; this also sidesteps the well-known
   inefficiency of cuBLAS strided-batched GEMM on very small matrices.

2. **Match eager FP32 accuracy.** Every ``tl.dot`` uses
   ``input_precision="ieee"`` so the contraction runs in true IEEE FP32 (no
   TF32). This keeps the potential-energy surface smooth.

3. **Compose with SeZM's ``make_fx`` lowering.** The operators are functional
   ``torch.library.custom_op`` instances (``mutates_args=()``) with registered
   fake kernels and autograd formulas. The backward is itself expressed as
   functional custom ops, so ``make_fx(tracing_mode="symbolic")`` can capture the
   energy path together with the force autograd graph used by inference.

Shapes / dtypes
---------------
``x``/``x_local`` and ``wigner`` are float tensors; fp32 is the supported
precision for the smooth potential-energy surface, while fp16/bf16 inputs
accumulate in fp32. ``src`` and ``coeff_index`` are int64 tensors. ``E`` (edges)
may exceed 2**31 elements once multiplied by the per-edge matrix size, so all
kernels use int64 addressing.
"""

from __future__ import (
    annotations,
)

import math

import torch
from torch import (
    Tensor,
)

from ..indexing import (
    build_m_major_index,
)

__all__ = [
    "TRITON_ROTATION_AVAILABLE",
    "rotate_back_block",
    "rotate_back_dense",
    "rotate_back_reference",
    "rotate_to_local_block",
    "rotate_to_local_dense",
    "rotate_to_local_reference",
]

try:
    import triton
    import triton.language as tl

    TRITON_ROTATION_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    TRITON_ROTATION_AVAILABLE = False


# ======================================================================
# Eager reference / fallback implementations
# ======================================================================
def rotate_to_local_reference(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """Eager ground-truth for ``rotate_to_local`` (``bmm(D_to_m, x[src])``)."""
    d_to_m = wigner[:, :dim_full, :dim_full].index_select(1, coeff_index)
    return torch.bmm(d_to_m, x.index_select(0, src))


def rotate_back_reference(
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """Eager ground-truth for ``rotate_back`` (``bmm(Dt_from_m, x_local)``)."""
    dt_from_m = wigner[:, :dim_full, :dim_full].index_select(2, coeff_index)
    return torch.bmm(dt_from_m, x_local)


def _rotate_to_local_bwd_eager(
    grad_out: Tensor,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> tuple[Tensor, Tensor]:
    """Eager backward of ``rotate_to_local`` returning ``(grad_x, grad_wigner)``."""
    w_rows = wigner[:, :dim_full, :dim_full].index_select(1, coeff_index)  # (E,Dm,D)
    x_src = x.index_select(0, src)  # (E,D,C)
    grad_x_src = torch.bmm(w_rows.transpose(1, 2), grad_out)  # (E,D,C)
    grad_x = torch.zeros_like(x).index_add_(0, src, grad_x_src)
    grad_rows = torch.bmm(grad_out, x_src.transpose(1, 2))  # (E,Dm,D)
    grad_block = torch.zeros(
        grad_out.shape[0], dim_full, dim_full, dtype=wigner.dtype, device=wigner.device
    )
    grad_block.index_copy_(1, coeff_index, grad_rows)
    grad_wigner = torch.zeros_like(wigner)
    grad_wigner[:, :dim_full, :dim_full] = grad_block
    return grad_x, grad_wigner


def _rotate_back_bwd_eager(
    grad_out: Tensor,
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> tuple[Tensor, Tensor]:
    """Eager backward of ``rotate_back`` returning ``(grad_x_local, grad_wigner)``."""
    w_cols = wigner[:, :dim_full, :dim_full].index_select(2, coeff_index)  # (E,D,Dm)
    grad_x_local = torch.bmm(w_cols.transpose(1, 2), grad_out)  # (E,Dm,C)
    grad_cols = torch.bmm(grad_out, x_local.transpose(1, 2))  # (E,D,Dm)
    grad_block = torch.zeros(
        grad_out.shape[0], dim_full, dim_full, dtype=wigner.dtype, device=wigner.device
    )
    grad_block.index_copy_(2, coeff_index, grad_cols)
    grad_wigner = torch.zeros_like(wigner)
    grad_wigner[:, :dim_full, :dim_full] = grad_block
    return grad_x_local, grad_wigner


# ======================================================================
# Tile-size helpers and autotuning configs
# ======================================================================
def _tile_dim(value: int) -> int:
    """Pick a single-tile edge: the next power of two, at least 16.

    Tiles spanning a whole dimension (the non-tiled ``N`` axis and the static
    ``BLOCK_N``) must be a power of two (``tl.arange``) *and* a multiple of 16
    (``tl.dot``); powers of two ``>= 16`` satisfy both. Packed dims map as
    ``16 -> 16`` (lmax 3), ``36 -> 64`` (lmax 5), ``64 -> 64`` (lmax 7),
    ``121 -> 128`` (lmax 10), ``C=64 -> 64``.
    """
    tile = 16
    target = max(int(value), 1)
    while tile < target:
        tile *= 2
    return tile


def _autotune_configs() -> list:
    """A small curated set of (BLOCK_M, BLOCK_K, num_warps, num_stages) configs.

    The per-edge GEMMs are tiny (M, K, N <= 128). We tile the output-row axis
    ``M`` across the grid and stream the contraction axis ``K`` in a pipelined
    loop, so the dominant Wigner load overlaps with the matmul. Autotuning over
    a handful of shapes lets one source kernel serve lmax 3..10 well (small
    tiles for lmax 3, larger tiles / more warps for lmax 10).
    """
    return [
        # Tiny tiles: best for lmax 3 (D=16), where a single 16x16 row tile and a
        # one-shot K step behave like a per-edge matvec with minimal overhead.
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 16}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ]


if TRITON_ROTATION_AVAILABLE:
    _CONFIGS = _autotune_configs()
    _KEY = ["dim_full", "reduced_dim", "channels"]

    # Block-diagonal kernels are fully unrolled over l (LMAX constexpr) and over
    # each l-block, with channels vectorized -- there is no GEMM tile to tune, so
    # we only sweep the warp count / pipeline depth, keyed on the channel width.
    _BD_CONFIGS = [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    _BD_KEY = ["channels"]

    # ==================================================================
    # Triton kernels
    #
    # Every kernel is one fused-gather GEMM ``C_out = A @ B`` with:
    #   * grid = (edge, ceil(M / BLOCK_M)) -- one program per (edge, row-tile),
    #   * a pipelined K-loop streaming BLOCK_K of the contraction at a time,
    #   * the Wigner row/column gather (by ``coeff_index``) and the node-feature
    #     gather (by ``src``) folded into the pointer arithmetic, so neither
    #     ``D_to_m``/``Dt_from_m`` nor ``x[src]`` is ever materialized.
    # All stores overwrite their tile (idempotent), which keeps autotuning safe.
    # ==================================================================
    @triton.autotune(configs=_CONFIGS, key=_KEY)
    @triton.jit
    def _to_local_fwd_kernel(
        x_ptr,
        src_ptr,
        w_ptr,
        idx_ptr,
        out_ptr,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        x_sn,
        x_sd,
        x_sc,
        w_se,
        w_sr,
        w_sk,
        o_se,
        o_sr,
        o_sc,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """``out[e,m,c] = sum_k W[e, coeff[m], k] * x[src[e], k, c]`` (M=Dm,K=D,N=C)."""
        edge = tl.program_id(0).to(tl.int64)
        row = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)  # over Dm
        chan = tl.arange(0, BLOCK_N)  # over C
        row_mask = row < reduced_dim
        chan_mask = chan < channels

        src_idx = tl.load(src_ptr + edge).to(tl.int64)
        coeff_rows = tl.load(idx_ptr + row, mask=row_mask, other=0).to(tl.int64)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(dim_full, BLOCK_K)):
            kk = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # over D
            k_mask = kk < dim_full
            w_tile = tl.load(
                w_ptr + edge * w_se + coeff_rows[:, None] * w_sr + kk[None, :] * w_sk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )  # (BLOCK_M, BLOCK_K) = W[coeff[m], k]
            x_tile = tl.load(
                x_ptr + src_idx * x_sn + kk[:, None] * x_sd + chan[None, :] * x_sc,
                mask=k_mask[:, None] & chan_mask[None, :],
                other=0.0,
            )  # (BLOCK_K, BLOCK_N) = x[src, k, c]
            acc = tl.dot(w_tile.to(x_tile.dtype), x_tile, acc, input_precision="ieee")

        tl.store(
            out_ptr + edge * o_se + row[:, None] * o_sr + chan[None, :] * o_sc,
            acc.to(out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & chan_mask[None, :],
        )

    @triton.autotune(configs=_CONFIGS, key=_KEY, reset_to_zero=["gx_ptr"])
    @triton.jit
    def _to_local_bwd_dx_kernel(
        go_ptr,
        src_ptr,
        w_ptr,
        idx_ptr,
        gx_ptr,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        go_se,
        go_sr,
        go_sc,
        w_se,
        w_sr,
        w_sk,
        gx_sn,
        gx_sd,
        gx_sc,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """``grad_x[src[e],d,c] += sum_m W[e, coeff[m], d] * grad_out[e,m,c]``.

        (M=D, K=Dm, N=C). The per-edge source gradient is atomically scattered
        straight into the zero-initialized ``grad_x`` (no ``x[src]``-sized
        scratch). ``reset_to_zero`` keeps the autotuner's trial runs from
        polluting the accumulator.
        """
        edge = tl.program_id(0).to(tl.int64)
        drow = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)  # over D
        chan = tl.arange(0, BLOCK_N)  # over C
        d_mask = drow < dim_full
        chan_mask = chan < channels

        src_idx = tl.load(src_ptr + edge).to(tl.int64)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(reduced_dim, BLOCK_K)):
            mm = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # over Dm
            m_mask = mm < reduced_dim
            coeff = tl.load(idx_ptr + mm, mask=m_mask, other=0).to(tl.int64)
            w_tile = tl.load(
                w_ptr + edge * w_se + coeff[None, :] * w_sr + drow[:, None] * w_sk,
                mask=d_mask[:, None] & m_mask[None, :],
                other=0.0,
            )  # (BLOCK_M(d), BLOCK_K(m)) = W[coeff[m], d]
            go_tile = tl.load(
                go_ptr + edge * go_se + mm[:, None] * go_sr + chan[None, :] * go_sc,
                mask=m_mask[:, None] & chan_mask[None, :],
                other=0.0,
            )  # (BLOCK_K(m), BLOCK_N(c))
            acc = tl.dot(w_tile.to(go_tile.dtype), go_tile, acc, input_precision="ieee")

        tl.atomic_add(
            gx_ptr + src_idx * gx_sn + drow[:, None] * gx_sd + chan[None, :] * gx_sc,
            acc,
            mask=d_mask[:, None] & chan_mask[None, :],
        )

    @triton.autotune(configs=_CONFIGS, key=_KEY)
    @triton.jit
    def _to_local_bwd_dw_kernel(
        go_ptr,
        x_ptr,
        src_ptr,
        idx_ptr,
        gw_ptr,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        go_se,
        go_sr,
        go_sc,
        x_sn,
        x_sd,
        x_sc,
        gw_se,
        gw_sr,
        gw_sk,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """``grad_W[e, coeff[m], d] = sum_c grad_out[e,m,c] * x[src[e], d, c]``.

        (M=Dm, K=C, N=D). Writes directly into rows ``coeff_index`` of the
        zero-initialized ``grad_wigner``.
        """
        edge = tl.program_id(0).to(tl.int64)
        mrow = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)  # over Dm
        dcol = tl.arange(0, BLOCK_N)  # over D
        m_mask = mrow < reduced_dim
        d_mask = dcol < dim_full

        coeff = tl.load(idx_ptr + mrow, mask=m_mask, other=0).to(tl.int64)
        src_idx = tl.load(src_ptr + edge).to(tl.int64)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(channels, BLOCK_K)):
            cc = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # over C
            c_mask = cc < channels
            go_tile = tl.load(
                go_ptr + edge * go_se + mrow[:, None] * go_sr + cc[None, :] * go_sc,
                mask=m_mask[:, None] & c_mask[None, :],
                other=0.0,
            )  # (BLOCK_M(m), BLOCK_K(c))
            x_tile = tl.load(
                x_ptr + src_idx * x_sn + dcol[None, :] * x_sd + cc[:, None] * x_sc,
                mask=c_mask[:, None] & d_mask[None, :],
                other=0.0,
            )  # (BLOCK_K(c), BLOCK_N(d)) = x[src, d, c]
            acc = tl.dot(go_tile.to(x_tile.dtype), x_tile, acc, input_precision="ieee")

        tl.store(
            gw_ptr + edge * gw_se + coeff[:, None] * gw_sr + dcol[None, :] * gw_sk,
            acc.to(gw_ptr.dtype.element_ty),
            mask=m_mask[:, None] & d_mask[None, :],
        )

    # ``rotate_back`` reads the Wigner *columns* selected by ``coeff_index``.
    # Gathering columns of a row-major ``(E, D, D)`` tensor is uncoalesced, so
    # instead we read *dense* Wigner rows (coalesced last axis) and gather /
    # scatter the small ``x_local`` through the inverse permutation
    # ``inv[k] = m  if coeff[m]==k else -1``. For ``mmax==lmax`` (a full
    # permutation) this is the same flop count with far better memory behaviour.
    @triton.autotune(configs=_CONFIGS, key=_KEY)
    @triton.jit
    def _back_fwd_kernel(
        xl_ptr,
        w_ptr,
        inv_ptr,
        out_ptr,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        xl_se,
        xl_sr,
        xl_sc,
        w_se,
        w_sr,
        w_sk,
        o_se,
        o_sd,
        o_sc,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """``out[e,d,c] = sum_k W[e,d,k] * x_local[e, inv[k], c]`` (M=D, K=D, N=C)."""
        edge = tl.program_id(0).to(tl.int64)
        drow = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)  # over D
        chan = tl.arange(0, BLOCK_N)  # over C
        d_mask = drow < dim_full
        chan_mask = chan < channels

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(dim_full, BLOCK_K)):
            kk = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # over D (contraction)
            k_mask = kk < dim_full
            inv_k = tl.load(inv_ptr + kk, mask=k_mask, other=-1).to(tl.int64)
            keep = inv_k >= 0
            w_tile = tl.load(
                w_ptr + edge * w_se + drow[:, None] * w_sr + kk[None, :] * w_sk,
                mask=d_mask[:, None] & k_mask[None, :],
                other=0.0,
            )  # (BLOCK_M(d), BLOCK_K(k)) = W[d, k]  (k contiguous -> coalesced)
            xl_tile = tl.load(
                xl_ptr + edge * xl_se + inv_k[:, None] * xl_sr + chan[None, :] * xl_sc,
                mask=keep[:, None] & chan_mask[None, :],
                other=0.0,
            )  # (BLOCK_K(k), BLOCK_N(c)) = x_local[inv[k], c]
            acc = tl.dot(w_tile.to(xl_tile.dtype), xl_tile, acc, input_precision="ieee")

        tl.store(
            out_ptr + edge * o_se + drow[:, None] * o_sd + chan[None, :] * o_sc,
            acc.to(out_ptr.dtype.element_ty),
            mask=d_mask[:, None] & chan_mask[None, :],
        )

    @triton.autotune(configs=_CONFIGS, key=_KEY)
    @triton.jit
    def _back_bwd_dx_kernel(
        go_ptr,
        w_ptr,
        inv_ptr,
        gxl_ptr,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        go_se,
        go_sd,
        go_sc,
        w_se,
        w_sr,
        w_sk,
        gxl_se,
        gxl_sr,
        gxl_sc,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """``grad_x_local[e, inv[k], c] = sum_d W[e,d,k] * grad_out[e,d,c]``.

        (M=D, K=D, N=C). Computes the dense ``k``-indexed gradient with coalesced
        Wigner reads, then scatters each full row ``k`` into reduced row
        ``inv[k]`` of ``grad_x_local``.
        """
        edge = tl.program_id(0).to(tl.int64)
        krow = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)  # over D
        chan = tl.arange(0, BLOCK_N)  # over C
        k_mask = krow < dim_full
        chan_mask = chan < channels

        inv_k = tl.load(inv_ptr + krow, mask=k_mask, other=-1).to(tl.int64)
        keep = inv_k >= 0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(dim_full, BLOCK_K)):
            dd = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # over D (contraction)
            d_mask = dd < dim_full
            w_tile = tl.load(
                w_ptr + edge * w_se + dd[None, :] * w_sr + krow[:, None] * w_sk,
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )  # (BLOCK_M(k), BLOCK_K(d)) = W[d, k]  (k contiguous -> coalesced)
            go_tile = tl.load(
                go_ptr + edge * go_se + dd[:, None] * go_sd + chan[None, :] * go_sc,
                mask=d_mask[:, None] & chan_mask[None, :],
                other=0.0,
            )  # (BLOCK_K(d), BLOCK_N(c))
            acc = tl.dot(w_tile.to(go_tile.dtype), go_tile, acc, input_precision="ieee")

        tl.store(
            gxl_ptr + edge * gxl_se + inv_k[:, None] * gxl_sr + chan[None, :] * gxl_sc,
            acc.to(gxl_ptr.dtype.element_ty),
            mask=keep[:, None] & chan_mask[None, :],
        )

    @triton.autotune(configs=_CONFIGS, key=_KEY)
    @triton.jit
    def _back_bwd_dw_kernel(
        go_ptr,
        xl_ptr,
        inv_ptr,
        gw_ptr,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        go_se,
        go_sd,
        go_sc,
        xl_se,
        xl_sr,
        xl_sc,
        gw_se,
        gw_sr,
        gw_sk,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """``grad_W[e,d,k] = sum_c grad_out[e,d,c] * x_local[e, inv[k], c]``.

        (M=D, K=C, N=D). Writes the dense ``(D, D)`` block of ``grad_wigner``
        with a coalesced last axis; columns ``k`` not selected by ``coeff_index``
        receive zero (``inv[k] < 0``), matching the eager column gather.
        """
        edge = tl.program_id(0).to(tl.int64)
        drow = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)  # over D
        kcol = tl.arange(0, BLOCK_N)  # over D
        d_mask = drow < dim_full
        k_mask = kcol < dim_full

        inv_k = tl.load(inv_ptr + kcol, mask=k_mask, other=-1).to(tl.int64)
        keep = inv_k >= 0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(channels, BLOCK_K)):
            cc = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # over C (contraction)
            c_mask = cc < channels
            go_tile = tl.load(
                go_ptr + edge * go_se + drow[:, None] * go_sd + cc[None, :] * go_sc,
                mask=d_mask[:, None] & c_mask[None, :],
                other=0.0,
            )  # (BLOCK_M(d), BLOCK_K(c))
            xl_tile = tl.load(
                xl_ptr + edge * xl_se + inv_k[None, :] * xl_sr + cc[:, None] * xl_sc,
                mask=c_mask[:, None] & keep[None, :],
                other=0.0,
            )  # (BLOCK_K(c), BLOCK_N(k)) = x_local[inv[k], c]
            acc = tl.dot(
                go_tile.to(xl_tile.dtype), xl_tile, acc, input_precision="ieee"
            )

        tl.store(
            gw_ptr + edge * gw_se + drow[:, None] * gw_sr + kcol[None, :] * gw_sk,
            acc.to(gw_ptr.dtype.element_ty),
            mask=d_mask[:, None] & k_mask[None, :],
        )

    # ==================================================================
    # Block-diagonal kernels (mmax == 1, block-diagonal Wigner-D)
    #
    # The Wigner-D matrix is block-diagonal by degree ``l``: block ``l`` is the
    # ``(2l+1) x (2l+1)`` sub-matrix on rows/cols ``[l^2 : (l+1)^2]`` and every
    # off-(l-block) entry is exactly 0. With ``mmax == 1`` the reduced layout
    # keeps, per degree ``l``, the orders ``m in {0}`` (l == 0) or
    # ``{0, -1, +1}`` (l >= 1). Output coefficient ``(l, m)`` therefore contracts
    # ONLY over the ``2l+1`` inputs of block ``l`` -- never the full ``D``.
    #
    # The m-major reduced index and the packed Wigner row/col are pure functions
    # of ``(l, m, LMAX)``::
    #
    #     reduced index:  m=0 -> l,  m=-1 -> LMAX+l,  m=+1 -> 2*LMAX+l
    #     packed (l, m):  l^2 + l + m   (so m=0 -> l^2+l, m=-1 -> -1, m=+1 -> +1)
    #
    # so the kernels need no ``coeff_index`` tensor: with ``LMAX`` a constexpr we
    # fully unroll over ``l`` and over each block, contracting exactly the
    # structural non-zeros (no padding, no wasted FLOPs). Channels are the
    # vectorized axis (``BLOCK_C`` spans the full width ``C``), so the backward
    # Wigner gradient is a single in-program ``tl.sum`` over channels.
    @triton.autotune(configs=_BD_CONFIGS, key=_BD_KEY)
    @triton.jit
    def _bd_to_local_fwd_kernel(
        x_ptr,
        src_ptr,
        w_ptr,
        out_ptr,
        n_edge,
        channels,
        x_sn,
        x_sd,
        x_sc,
        w_se,
        w_sr,
        w_sk,
        o_se,
        o_sr,
        o_sc,
        LMAX: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """``out[e,(l,m),c] = sum_{j} W[e, l^2+l+m, l^2+j] * x[src[e], l^2+j, c]``."""
        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < channels
        src_idx = tl.load(src_ptr + edge).to(tl.int64)

        for l in tl.static_range(0, LMAX + 1):
            base = l * l
            r0 = base + l  # packed row of order m=0
            acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc_m = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc_p = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for j in tl.static_range(0, 2 * l + 1):
                col = base + j
                x_vec = tl.load(
                    x_ptr + src_idx * x_sn + col * x_sd + chan * x_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                acc0 += tl.load(w_ptr + edge * w_se + r0 * w_sr + col * w_sk) * x_vec
                if l >= 1:
                    acc_m += (
                        tl.load(w_ptr + edge * w_se + (r0 - 1) * w_sr + col * w_sk)
                        * x_vec
                    )
                    acc_p += (
                        tl.load(w_ptr + edge * w_se + (r0 + 1) * w_sr + col * w_sk)
                        * x_vec
                    )
            tl.store(
                out_ptr + edge * o_se + l * o_sr + chan * o_sc,
                acc0.to(out_ptr.dtype.element_ty),
                mask=cmask,
            )
            if l >= 1:
                tl.store(
                    out_ptr + edge * o_se + (LMAX + l) * o_sr + chan * o_sc,
                    acc_m.to(out_ptr.dtype.element_ty),
                    mask=cmask,
                )
                tl.store(
                    out_ptr + edge * o_se + (2 * LMAX + l) * o_sr + chan * o_sc,
                    acc_p.to(out_ptr.dtype.element_ty),
                    mask=cmask,
                )

    @triton.autotune(configs=_BD_CONFIGS, key=_BD_KEY, reset_to_zero=["gx_ptr"])
    @triton.jit
    def _bd_to_local_bwd_kernel(
        go_ptr,
        x_ptr,
        src_ptr,
        w_ptr,
        gx_ptr,
        gw_ptr,
        n_edge,
        channels,
        go_se,
        go_sr,
        go_sc,
        x_sn,
        x_sd,
        x_sc,
        w_se,
        w_sr,
        w_sk,
        gx_sn,
        gx_sd,
        gx_sc,
        gw_se,
        gw_sr,
        gw_sk,
        LMAX: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Fused block-diagonal backward of ``rotate_to_local``.

        Per edge (full channel width in one program): scatters
        ``grad_x[src, l^2+j, :] += sum_m W[l^2+l+m, l^2+j] * grad_out[(l,m), :]``
        and writes ``grad_W[l^2+l+m, l^2+j] = sum_c grad_out[(l,m),c] * x[l^2+j,c]``
        for the structural non-zeros only.
        """
        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < channels
        src_idx = tl.load(src_ptr + edge).to(tl.int64)

        for l in tl.static_range(0, LMAX + 1):
            base = l * l
            r0 = base + l
            go0 = tl.load(
                go_ptr + edge * go_se + l * go_sr + chan * go_sc,
                mask=cmask,
                other=0.0,
            ).to(tl.float32)
            if l >= 1:
                go_m = tl.load(
                    go_ptr + edge * go_se + (LMAX + l) * go_sr + chan * go_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                go_p = tl.load(
                    go_ptr + edge * go_se + (2 * LMAX + l) * go_sr + chan * go_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
            for j in tl.static_range(0, 2 * l + 1):
                col = base + j
                x_vec = tl.load(
                    x_ptr + src_idx * x_sn + col * x_sd + chan * x_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                w0 = tl.load(w_ptr + edge * w_se + r0 * w_sr + col * w_sk)
                gx_row = w0 * go0
                tl.store(
                    gw_ptr + edge * gw_se + r0 * gw_sr + col * gw_sk,
                    tl.sum(go0 * x_vec).to(gw_ptr.dtype.element_ty),
                )
                if l >= 1:
                    wm = tl.load(w_ptr + edge * w_se + (r0 - 1) * w_sr + col * w_sk)
                    wp = tl.load(w_ptr + edge * w_se + (r0 + 1) * w_sr + col * w_sk)
                    gx_row += wm * go_m + wp * go_p
                    tl.store(
                        gw_ptr + edge * gw_se + (r0 - 1) * gw_sr + col * gw_sk,
                        tl.sum(go_m * x_vec).to(gw_ptr.dtype.element_ty),
                    )
                    tl.store(
                        gw_ptr + edge * gw_se + (r0 + 1) * gw_sr + col * gw_sk,
                        tl.sum(go_p * x_vec).to(gw_ptr.dtype.element_ty),
                    )
                tl.atomic_add(
                    gx_ptr + src_idx * gx_sn + col * gx_sd + chan * gx_sc,
                    gx_row,
                    mask=cmask,
                )

    @triton.autotune(configs=_BD_CONFIGS, key=_BD_KEY)
    @triton.jit
    def _bd_back_fwd_kernel(
        xl_ptr,
        w_ptr,
        out_ptr,
        n_edge,
        channels,
        xl_se,
        xl_sr,
        xl_sc,
        w_se,
        w_sr,
        w_sk,
        o_se,
        o_sd,
        o_sc,
        LMAX: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """``out[e, l^2+j, c] = sum_m W[e, l^2+j, l^2+l+m] * x_local[(l,m), c]``."""
        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < channels

        for l in tl.static_range(0, LMAX + 1):
            base = l * l
            r0 = base + l  # packed col of order m=0
            xl0 = tl.load(
                xl_ptr + edge * xl_se + l * xl_sr + chan * xl_sc,
                mask=cmask,
                other=0.0,
            ).to(tl.float32)
            if l >= 1:
                xl_m = tl.load(
                    xl_ptr + edge * xl_se + (LMAX + l) * xl_sr + chan * xl_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                xl_p = tl.load(
                    xl_ptr + edge * xl_se + (2 * LMAX + l) * xl_sr + chan * xl_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
            for j in tl.static_range(0, 2 * l + 1):
                d = base + j  # full packed output row
                acc = tl.load(w_ptr + edge * w_se + d * w_sr + r0 * w_sk) * xl0
                if l >= 1:
                    acc += (
                        tl.load(w_ptr + edge * w_se + d * w_sr + (r0 - 1) * w_sk) * xl_m
                    )
                    acc += (
                        tl.load(w_ptr + edge * w_se + d * w_sr + (r0 + 1) * w_sk) * xl_p
                    )
                tl.store(
                    out_ptr + edge * o_se + d * o_sd + chan * o_sc,
                    acc.to(out_ptr.dtype.element_ty),
                    mask=cmask,
                )

    @triton.autotune(configs=_BD_CONFIGS, key=_BD_KEY)
    @triton.jit
    def _bd_back_bwd_kernel(
        go_ptr,
        xl_ptr,
        w_ptr,
        gxl_ptr,
        gw_ptr,
        n_edge,
        channels,
        go_se,
        go_sd,
        go_sc,
        xl_se,
        xl_sr,
        xl_sc,
        w_se,
        w_sr,
        w_sk,
        gxl_se,
        gxl_sr,
        gxl_sc,
        gw_se,
        gw_sr,
        gw_sk,
        LMAX: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Fused block-diagonal backward of ``rotate_back``.

        Per edge (full channel width in one program): writes
        ``grad_x_local[(l,m), :] = sum_j W[l^2+j, l^2+l+m] * grad_out[l^2+j, :]``
        (no scatter -- ``x_local`` is per-edge) and
        ``grad_W[l^2+j, l^2+l+m] = sum_c grad_out[l^2+j, c] * x_local[(l,m), c]``.
        """
        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < channels

        for l in tl.static_range(0, LMAX + 1):
            base = l * l
            r0 = base + l  # packed col of order m=0
            xl0 = tl.load(
                xl_ptr + edge * xl_se + l * xl_sr + chan * xl_sc,
                mask=cmask,
                other=0.0,
            ).to(tl.float32)
            gxl0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            if l >= 1:
                xl_m = tl.load(
                    xl_ptr + edge * xl_se + (LMAX + l) * xl_sr + chan * xl_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                xl_p = tl.load(
                    xl_ptr + edge * xl_se + (2 * LMAX + l) * xl_sr + chan * xl_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                gxl_m = tl.zeros((BLOCK_C,), dtype=tl.float32)
                gxl_p = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for j in tl.static_range(0, 2 * l + 1):
                d = base + j  # full packed row (output of forward / grad_out row)
                go_d = tl.load(
                    go_ptr + edge * go_se + d * go_sd + chan * go_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                w0 = tl.load(w_ptr + edge * w_se + d * w_sr + r0 * w_sk)
                gxl0 += w0 * go_d
                tl.store(
                    gw_ptr + edge * gw_se + d * gw_sr + r0 * gw_sk,
                    tl.sum(go_d * xl0).to(gw_ptr.dtype.element_ty),
                )
                if l >= 1:
                    wm = tl.load(w_ptr + edge * w_se + d * w_sr + (r0 - 1) * w_sk)
                    wp = tl.load(w_ptr + edge * w_se + d * w_sr + (r0 + 1) * w_sk)
                    gxl_m += wm * go_d
                    gxl_p += wp * go_d
                    tl.store(
                        gw_ptr + edge * gw_se + d * gw_sr + (r0 - 1) * gw_sk,
                        tl.sum(go_d * xl_m).to(gw_ptr.dtype.element_ty),
                    )
                    tl.store(
                        gw_ptr + edge * gw_se + d * gw_sr + (r0 + 1) * gw_sk,
                        tl.sum(go_d * xl_p).to(gw_ptr.dtype.element_ty),
                    )
            tl.store(
                gxl_ptr + edge * gxl_se + l * gxl_sr + chan * gxl_sc,
                gxl0.to(gxl_ptr.dtype.element_ty),
                mask=cmask,
            )
            if l >= 1:
                tl.store(
                    gxl_ptr + edge * gxl_se + (LMAX + l) * gxl_sr + chan * gxl_sc,
                    gxl_m.to(gxl_ptr.dtype.element_ty),
                    mask=cmask,
                )
                tl.store(
                    gxl_ptr + edge * gxl_se + (2 * LMAX + l) * gxl_sr + chan * gxl_sc,
                    gxl_p.to(gxl_ptr.dtype.element_ty),
                    mask=cmask,
                )


# ======================================================================
# Triton launch wrappers
# ======================================================================
def _grid_over_rows(n_edge: int, rows: int):
    """Grid callable: one program per (edge, BLOCK_M-sized row tile)."""
    return lambda meta: (n_edge, triton.cdiv(rows, meta["BLOCK_M"]))


def _inverse_index(coeff_index: Tensor, dim_full: int) -> Tensor:
    """Inverse permutation ``inv[k] = m`` where ``coeff_index[m] == k`` else ``-1``.

    Maps a full packed position ``k`` back to its reduced-layout slot. Used by the
    ``rotate_back`` kernels so they can read dense Wigner rows (coalesced) and
    gather/scatter the small ``x_local`` instead of gathering Wigner columns.
    """
    inv = torch.full((int(dim_full),), -1, dtype=torch.int64, device=coeff_index.device)
    inv[coeff_index] = torch.arange(
        coeff_index.numel(), dtype=torch.int64, device=coeff_index.device
    )
    return inv


def _launch_rotate_to_local_fwd(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    n_edge = int(src.shape[0])
    reduced_dim = int(coeff_index.shape[0])
    channels = int(x.shape[2])
    out = torch.empty((n_edge, reduced_dim, channels), dtype=x.dtype, device=x.device)
    if n_edge == 0:
        return out
    _to_local_fwd_kernel[_grid_over_rows(n_edge, reduced_dim)](
        x,
        src,
        wigner,
        coeff_index,
        out,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=_tile_dim(channels),
    )
    return out


def _launch_rotate_to_local_bwd(
    grad_out: Tensor,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> tuple[Tensor, Tensor]:
    n_edge = int(src.shape[0])
    reduced_dim = int(coeff_index.shape[0])
    channels = int(x.shape[2])
    grad_x = torch.zeros_like(x)
    grad_wigner = torch.zeros_like(wigner)
    if n_edge == 0:
        return grad_x, grad_wigner

    # --- grad_x: per-edge GEMM atomically scattered into grad_x by src ---
    _to_local_bwd_dx_kernel[_grid_over_rows(n_edge, dim_full)](
        grad_out,
        src,
        wigner,
        coeff_index,
        grad_x,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_x.stride(2),
        BLOCK_N=_tile_dim(channels),
    )

    # --- grad_wigner: per-edge GEMM written into rows ``coeff_index`` ---
    _to_local_bwd_dw_kernel[_grid_over_rows(n_edge, reduced_dim)](
        grad_out,
        x,
        src,
        coeff_index,
        grad_wigner,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        grad_wigner.stride(0),
        grad_wigner.stride(1),
        grad_wigner.stride(2),
        BLOCK_N=_tile_dim(dim_full),
    )
    return grad_x, grad_wigner


def _launch_rotate_back_fwd(
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    n_edge = int(x_local.shape[0])
    reduced_dim = int(coeff_index.shape[0])
    channels = int(x_local.shape[2])
    out = torch.empty(
        (n_edge, dim_full, channels), dtype=x_local.dtype, device=x_local.device
    )
    if n_edge == 0:
        return out
    inv_index = _inverse_index(coeff_index, dim_full)
    _back_fwd_kernel[_grid_over_rows(n_edge, dim_full)](
        x_local,
        wigner,
        inv_index,
        out,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=_tile_dim(channels),
    )
    return out


def _launch_rotate_back_bwd(
    grad_out: Tensor,
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> tuple[Tensor, Tensor]:
    n_edge = int(x_local.shape[0])
    reduced_dim = int(coeff_index.shape[0])
    channels = int(x_local.shape[2])
    grad_x_local = torch.empty_like(x_local)
    grad_wigner = torch.zeros_like(wigner)
    if n_edge == 0:
        return grad_x_local, grad_wigner

    inv_index = _inverse_index(coeff_index, dim_full)
    _back_bwd_dx_kernel[_grid_over_rows(n_edge, dim_full)](
        grad_out,
        wigner,
        inv_index,
        grad_x_local,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        grad_x_local.stride(0),
        grad_x_local.stride(1),
        grad_x_local.stride(2),
        BLOCK_N=_tile_dim(channels),
    )
    _back_bwd_dw_kernel[_grid_over_rows(n_edge, dim_full)](
        grad_out,
        x_local,
        inv_index,
        grad_wigner,
        n_edge,
        reduced_dim,
        dim_full,
        channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        grad_wigner.stride(0),
        grad_wigner.stride(1),
        grad_wigner.stride(2),
        BLOCK_N=_tile_dim(dim_full),
    )
    return grad_x_local, grad_wigner


# ======================================================================
# Block-diagonal launch wrappers + layout detection (mmax == 1)
# ======================================================================
def _block_layout_lmax(coeff_index: Tensor, dim_full: int) -> int:
    """Return ``lmax`` if ``(coeff_index, dim_full)`` is the m-major ``mmax=1``
    layout that the block-diagonal kernels assume, else ``-1``.

    This intentionally checks only shape-level invariants.  The block kernels
    ignore ``coeff_index`` values, so production callers must only use the block
    entry points when they own the canonical m-major ``mmax=1`` index.
    """
    dim_full = int(dim_full)
    root = math.isqrt(dim_full)
    if root * root != dim_full:
        return -1
    lmax = root - 1
    try:
        numel = int(coeff_index.shape[0])
    except Exception:  # pragma: no cover - exotic shape proxies
        return -1
    if lmax < 1 or numel != 3 * lmax + 1:
        return -1
    return lmax


def _launch_bd_to_local_fwd(
    x: Tensor, src: Tensor, wigner: Tensor, lmax: int
) -> Tensor:
    n_edge = int(src.shape[0])
    channels = int(x.shape[2])
    out = torch.empty((n_edge, 3 * lmax + 1, channels), dtype=x.dtype, device=x.device)
    if n_edge == 0:
        return out
    _bd_to_local_fwd_kernel[(n_edge,)](
        x,
        src,
        wigner,
        out,
        n_edge,
        channels,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LMAX=lmax,
        BLOCK_C=_tile_dim(channels),
    )
    return out


def _launch_bd_to_local_bwd(
    grad_out: Tensor, x: Tensor, src: Tensor, wigner: Tensor, lmax: int
) -> tuple[Tensor, Tensor]:
    n_edge = int(src.shape[0])
    channels = int(x.shape[2])
    grad_x = torch.zeros_like(x)
    grad_wigner = torch.zeros_like(wigner)
    if n_edge == 0:
        return grad_x, grad_wigner
    _bd_to_local_bwd_kernel[(n_edge,)](
        grad_out,
        x,
        src,
        wigner,
        grad_x,
        grad_wigner,
        n_edge,
        channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_x.stride(2),
        grad_wigner.stride(0),
        grad_wigner.stride(1),
        grad_wigner.stride(2),
        LMAX=lmax,
        BLOCK_C=_tile_dim(channels),
    )
    return grad_x, grad_wigner


def _launch_bd_back_fwd(x_local: Tensor, wigner: Tensor, lmax: int) -> Tensor:
    n_edge = int(x_local.shape[0])
    channels = int(x_local.shape[2])
    dim_full = (lmax + 1) ** 2
    out = torch.empty(
        (n_edge, dim_full, channels), dtype=x_local.dtype, device=x_local.device
    )
    if n_edge == 0:
        return out
    _bd_back_fwd_kernel[(n_edge,)](
        x_local,
        wigner,
        out,
        n_edge,
        channels,
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LMAX=lmax,
        BLOCK_C=_tile_dim(channels),
    )
    return out


def _launch_bd_back_bwd(
    grad_out: Tensor, x_local: Tensor, wigner: Tensor, lmax: int
) -> tuple[Tensor, Tensor]:
    n_edge = int(x_local.shape[0])
    channels = int(x_local.shape[2])
    grad_x_local = torch.empty_like(x_local)
    grad_wigner = torch.zeros_like(wigner)
    if n_edge == 0:
        return grad_x_local, grad_wigner
    _bd_back_bwd_kernel[(n_edge,)](
        grad_out,
        x_local,
        wigner,
        grad_x_local,
        grad_wigner,
        n_edge,
        channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        wigner.stride(0),
        wigner.stride(1),
        wigner.stride(2),
        grad_x_local.stride(0),
        grad_x_local.stride(1),
        grad_x_local.stride(2),
        grad_wigner.stride(0),
        grad_wigner.stride(1),
        grad_wigner.stride(2),
        LMAX=lmax,
        BLOCK_C=_tile_dim(channels),
    )
    return grad_x_local, grad_wigner


# ======================================================================
# Dispatch helpers (triton on CUDA float, eager otherwise)
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        TRITON_ROTATION_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _rotate_to_local_impl(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    if not _use_triton(x):
        return rotate_to_local_reference(x, src, wigner, coeff_index, dim_full)
    return _launch_rotate_to_local_fwd(
        x, src.contiguous(), wigner, coeff_index.contiguous(), int(dim_full)
    )


def _rotate_to_local_bwd_impl(
    grad_out: Tensor,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(x):
        return _rotate_to_local_bwd_eager(
            grad_out, x, src, wigner, coeff_index, dim_full
        )
    return _launch_rotate_to_local_bwd(
        grad_out.contiguous(),
        x,
        src.contiguous(),
        wigner,
        coeff_index.contiguous(),
        int(dim_full),
    )


def _rotate_back_impl(
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    if not _use_triton(x_local):
        return rotate_back_reference(x_local, wigner, coeff_index, dim_full)
    return _launch_rotate_back_fwd(
        x_local, wigner, coeff_index.contiguous(), int(dim_full)
    )


def _rotate_back_bwd_impl(
    grad_out: Tensor,
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(x_local):
        return _rotate_back_bwd_eager(grad_out, x_local, wigner, coeff_index, dim_full)
    return _launch_rotate_back_bwd(
        grad_out.contiguous(),
        x_local,
        wigner,
        coeff_index.contiguous(),
        int(dim_full),
    )


# --- block-diagonal impls (mmax == 1; assume block-diagonal Wigner-D) ---
def _block_rotate_to_local_impl(
    x: Tensor, src: Tensor, wigner: Tensor, lmax: int
) -> Tensor:
    if not _use_triton(x):
        coeff = build_m_major_index(int(lmax), 1, device=x.device)
        return rotate_to_local_reference(x, src, wigner, coeff, (int(lmax) + 1) ** 2)
    return _launch_bd_to_local_fwd(x, src.contiguous(), wigner, int(lmax))


def _block_rotate_to_local_bwd_impl(
    grad_out: Tensor, x: Tensor, src: Tensor, wigner: Tensor, lmax: int
) -> tuple[Tensor, Tensor]:
    if not _use_triton(x):
        coeff = build_m_major_index(int(lmax), 1, device=x.device)
        return _rotate_to_local_bwd_eager(
            grad_out, x, src, wigner, coeff, (int(lmax) + 1) ** 2
        )
    return _launch_bd_to_local_bwd(
        grad_out.contiguous(), x, src.contiguous(), wigner, int(lmax)
    )


def _block_rotate_back_impl(x_local: Tensor, wigner: Tensor, lmax: int) -> Tensor:
    if not _use_triton(x_local):
        coeff = build_m_major_index(int(lmax), 1, device=x_local.device)
        return rotate_back_reference(x_local, wigner, coeff, (int(lmax) + 1) ** 2)
    return _launch_bd_back_fwd(x_local, wigner, int(lmax))


def _block_rotate_back_bwd_impl(
    grad_out: Tensor, x_local: Tensor, wigner: Tensor, lmax: int
) -> tuple[Tensor, Tensor]:
    if not _use_triton(x_local):
        coeff = build_m_major_index(int(lmax), 1, device=x_local.device)
        return _rotate_back_bwd_eager(
            grad_out, x_local, wigner, coeff, (int(lmax) + 1) ** 2
        )
    return _launch_bd_back_bwd(grad_out.contiguous(), x_local, wigner, int(lmax))


# ======================================================================
# Modern functional custom ops + fake + autograd registration
# ======================================================================
# Forward and backward are both *functional* custom ops (mutates_args=()), so
# functionalization keeps the full gradient path -- including grad w.r.t.
# ``wigner`` -- intact under ``torch.compile``.

_rotate_to_local_op = torch.library.custom_op(
    "sezm_triton::rotate_to_local", mutates_args=()
)(_rotate_to_local_impl)

_rotate_to_local_bwd_op = torch.library.custom_op(
    "sezm_triton::rotate_to_local_bwd", mutates_args=()
)(_rotate_to_local_bwd_impl)

_rotate_back_op = torch.library.custom_op("sezm_triton::rotate_back", mutates_args=())(
    _rotate_back_impl
)

_rotate_back_bwd_op = torch.library.custom_op(
    "sezm_triton::rotate_back_bwd", mutates_args=()
)(_rotate_back_bwd_impl)


@_rotate_to_local_op.register_fake
def _(x, src, wigner, coeff_index, dim_full):
    return x.new_empty((src.shape[0], coeff_index.shape[0], x.shape[2]))


@_rotate_to_local_bwd_op.register_fake
def _(grad_out, x, src, wigner, coeff_index, dim_full):
    return torch.empty_like(x), torch.empty_like(wigner)


@_rotate_back_op.register_fake
def _(x_local, wigner, coeff_index, dim_full):
    return x_local.new_empty((x_local.shape[0], dim_full, x_local.shape[2]))


@_rotate_back_bwd_op.register_fake
def _(grad_out, x_local, wigner, coeff_index, dim_full):
    return torch.empty_like(x_local), torch.empty_like(wigner)


def _rotate_to_local_setup_context(ctx, inputs, output):
    x, src, wigner, coeff_index, dim_full = inputs
    ctx.save_for_backward(x, src, wigner, coeff_index)
    ctx.dim_full = dim_full


def _rotate_to_local_backward(ctx, grad_out):
    x, src, wigner, coeff_index = ctx.saved_tensors
    grad_x, grad_wigner = _rotate_to_local_bwd_op(
        grad_out, x, src, wigner, coeff_index, ctx.dim_full
    )
    return grad_x, None, grad_wigner, None, None


def _rotate_back_setup_context(ctx, inputs, output):
    x_local, wigner, coeff_index, dim_full = inputs
    ctx.save_for_backward(x_local, wigner, coeff_index)
    ctx.dim_full = dim_full


def _rotate_back_backward(ctx, grad_out):
    x_local, wigner, coeff_index = ctx.saved_tensors
    grad_x_local, grad_wigner = _rotate_back_bwd_op(
        grad_out, x_local, wigner, coeff_index, ctx.dim_full
    )
    return grad_x_local, grad_wigner, None, None


_rotate_to_local_op.register_autograd(
    _rotate_to_local_backward, setup_context=_rotate_to_local_setup_context
)
_rotate_back_op.register_autograd(
    _rotate_back_backward, setup_context=_rotate_back_setup_context
)


# --- block-diagonal custom ops (carry only ``lmax``; no coeff_index tensor) ---
_block_to_local_op = torch.library.custom_op(
    "sezm_triton::rotate_to_local_block", mutates_args=()
)(_block_rotate_to_local_impl)

_block_to_local_bwd_op = torch.library.custom_op(
    "sezm_triton::rotate_to_local_block_bwd", mutates_args=()
)(_block_rotate_to_local_bwd_impl)

_block_back_op = torch.library.custom_op(
    "sezm_triton::rotate_back_block", mutates_args=()
)(_block_rotate_back_impl)

_block_back_bwd_op = torch.library.custom_op(
    "sezm_triton::rotate_back_block_bwd", mutates_args=()
)(_block_rotate_back_bwd_impl)


@_block_to_local_op.register_fake
def _(x, src, wigner, lmax):
    return x.new_empty((src.shape[0], 3 * int(lmax) + 1, x.shape[2]))


@_block_to_local_bwd_op.register_fake
def _(grad_out, x, src, wigner, lmax):
    return torch.empty_like(x), torch.empty_like(wigner)


@_block_back_op.register_fake
def _(x_local, wigner, lmax):
    return x_local.new_empty((x_local.shape[0], (int(lmax) + 1) ** 2, x_local.shape[2]))


@_block_back_bwd_op.register_fake
def _(grad_out, x_local, wigner, lmax):
    return torch.empty_like(x_local), torch.empty_like(wigner)


def _block_to_local_setup_context(ctx, inputs, output):
    x, src, wigner, lmax = inputs
    ctx.save_for_backward(x, src, wigner)
    ctx.lmax = lmax


def _block_to_local_backward(ctx, grad_out):
    x, src, wigner = ctx.saved_tensors
    grad_x, grad_wigner = _block_to_local_bwd_op(grad_out, x, src, wigner, ctx.lmax)
    return grad_x, None, grad_wigner, None


def _block_back_setup_context(ctx, inputs, output):
    x_local, wigner, lmax = inputs
    ctx.save_for_backward(x_local, wigner)
    ctx.lmax = lmax


def _block_back_backward(ctx, grad_out):
    x_local, wigner = ctx.saved_tensors
    grad_x_local, grad_wigner = _block_back_bwd_op(grad_out, x_local, wigner, ctx.lmax)
    return grad_x_local, grad_wigner, None


_block_to_local_op.register_autograd(
    _block_to_local_backward, setup_context=_block_to_local_setup_context
)
_block_back_op.register_autograd(
    _block_back_backward, setup_context=_block_back_setup_context
)


# ======================================================================
# Public API
# ======================================================================
# --- Public entry points -----------------------------------------------------
def rotate_to_local_dense(
    x: Tensor, src: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
) -> Tensor:
    """Apply the general ``global -> local`` rotation.

    This entry point honors every value in ``coeff_index`` and supports any
    reduced coefficient layout.  It computes the same operation as
    ``rotate_to_local_reference`` while avoiding materialized gather operands on
    CUDA.
    """
    return _rotate_to_local_op(x, src, wigner, coeff_index, int(dim_full))


def rotate_back_dense(
    x_local: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
) -> Tensor:
    """Apply the general ``local -> global`` rotation.

    This entry point honors every value in ``coeff_index`` and supports any
    reduced coefficient layout.  It computes the same operation as
    ``rotate_back_reference`` while avoiding materialized gather operands on
    CUDA.
    """
    return _rotate_back_op(x_local, wigner, coeff_index, int(dim_full))


def rotate_to_local_block(
    x: Tensor, src: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
) -> Tensor:
    """Apply the block-diagonal ``global -> local`` rotation.

    Use this only when the caller owns the invariant that ``coeff_index`` is the
    canonical m-major ``mmax=1`` index produced by
    :func:`build_m_major_index`.  The kernel ignores the tensor values in
    ``coeff_index`` and derives the layout from ``lmax``.
    """
    lmax = _block_layout_lmax(coeff_index, dim_full)
    if lmax < 0:
        raise ValueError(
            "rotate_to_local_block requires the m-major mmax=1 coefficient layout."
        )
    return _block_to_local_op(x, src, wigner, lmax)


def rotate_back_block(
    x_local: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
) -> Tensor:
    """Apply the block-diagonal ``local -> global`` rotation.

    Use this only when the caller owns the invariant that ``coeff_index`` is the
    canonical m-major ``mmax=1`` index produced by
    :func:`build_m_major_index`.  The kernel ignores the tensor values in
    ``coeff_index`` and derives the layout from ``lmax``.
    """
    lmax = _block_layout_lmax(coeff_index, dim_full)
    if lmax < 0:
        raise ValueError(
            "rotate_back_block requires the m-major mmax=1 coefficient layout."
        )
    return _block_back_op(x_local, wigner, lmax)
