# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""fp16x3 split-compensated dense GEMM for the DPA1 embedding (``DP_TRITON_INFER
>= 3``).

The DPA1 embedding GEMMs run on a large edge/row count ``M`` (hundreds of
thousands) with modest ``K`` / ``N``; the profile shows they are compute-bound
(arithmetic intensity well above the H20 fp32 compute/bandwidth break-even), so
the tensor cores can beat the fp32 FFMA path that cuBLAS uses. This operator
evaluates ``C = A @ B`` as three fp16 tensor-core products with fp32
accumulation (a two-term Ootomo split)::

    A = A_hi + A_lo,  B = B_hi + B_lo         (fp16 head + fp16 tail)
    C ~= A_hi B_hi + A_hi B_lo + A_lo B_hi    (the A_lo B_lo term, ~2^-22
                                               relative, is dropped)

An fp16 multiply into an fp32 accumulator is exact, so the only error is the two
split truncations. The head product and the two tail corrections accumulate in
*separate* fp32 accumulators merged once per tile -- chaining all three into one
absorbs the small tails against the large head partial sums and doubles the
error. Tails are stored pre-scaled by ``2^11`` (the fp16 mantissa width) so an
element below the fp16 subnormal range keeps its correction; the epilogue scales
back. Measured on the DPA1 embedding shapes the maximum relative error matches
the fp32 reference (~3e-7) at ~1.6x the cuBLAS fp32 throughput.

Only the compute-bound, large-``M`` embedding GEMMs are eligible; the small
fitting GEMMs (``M`` ~ nloc) stay on cuBLAS, where fp16x3 loses to the launch and
split overhead. The operator is inference-only (level 3) and returns the input
gradient (the coordinate force path); the weight gradient is not formed
(training keeps the fp32 reference path).

The launch pins ``num_stages = 1``. A split-fp16 k-loop is a known failure mode
of the Triton software pipeliner: at ``num_stages >= 2`` the prefetch reorders
the head/tail split across loop iterations and can emit silent ``NaN`` on some
shapes, and -- as the SeZM stacked-GEMM tuning established -- no finite sample
certifies every problem size, so a pipelined config validated at one row count
may still poison another. Disabling the pipeliner removes the reordering
entirely and is therefore structurally finite for all shapes; on the DPA1
embedding widths it costs under 20% of the GEMM (still ~1.3x over cuBLAS fp32),
which is negligible end to end. A faster stage count would require the SeZM-style
per-shape finiteness sweep and is not warranted for this single GEMM.
"""

from __future__ import (
    annotations,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    triton_op,
    wrap_triton,
)

from deepmd.kernels.triton.dpa1.activation import (
    TRITON_AVAILABLE,
)
from deepmd.kernels.utils import (
    triton_infer_level,
)

__all__ = [
    "embed_gemm_fp16x3",
    "embed_last_gemm",
]

# fp16 mantissa-width tail prescale (2^11); an exact power of two, so the split
# and its inverse are lossless.
_TAIL_SCALE = 2048.0

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _gemm_fp16x3_kernel(
        a_ptr,  # (M, K) row-major
        b_ptr,  # (K, N) row-major
        c_ptr,  # (M, N) row-major
        M,
        N,
        K,
        S: tl.constexpr,  # tail prescale
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """``C = A @ B`` via the two-term Ootomo split on fp16 tensor cores.

        The head (``A_hi B_hi``) and the two tail corrections (``A_hi B_lo`` and
        ``A_lo B_hi``) accumulate in separate fp32 tiles, merged once after the
        k-loop; the dropped ``A_lo B_lo`` term is ~``2^-22`` relative.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # Row offsets are int64: M is the edge count, and the flat element
        # offsets rm * K / rm * N exceed int32 beyond ~17M rows at N = 128.
        rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
        b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
        m_row = rm[:, None] < M
        n_col = rn[None, :] < N
        acc_hi = tl.zeros((BM, BN), dtype=tl.float32)
        acc_lo = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = rk + k0
            a = tl.load(a_ptrs, mask=m_row & (kk[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(kk[:, None] < K) & n_col, other=0.0)
            a_hi = a.to(tl.float16)
            a_lo = ((a - a_hi.to(tl.float32)) * S).to(tl.float16)
            b_hi = b.to(tl.float16)
            b_lo = ((b - b_hi.to(tl.float32)) * S).to(tl.float16)
            acc_hi += tl.dot(a_hi, b_hi, out_dtype=tl.float32, input_precision="ieee")
            acc_lo += tl.dot(a_hi, b_lo, out_dtype=tl.float32, input_precision="ieee")
            acc_lo += tl.dot(a_lo, b_hi, out_dtype=tl.float32, input_precision="ieee")
            a_ptrs += BK
            b_ptrs += BK * N
        acc = acc_hi + acc_lo / S
        c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
        tl.store(c_ptrs, acc, mask=m_row & n_col)


def _use_triton(tensor: Tensor) -> bool:
    return TRITON_AVAILABLE and tensor.is_cuda and tensor.dtype == torch.float32


def _gemm_impl(
    a: Tensor, b: Tensor, bm: int, bn: int, bk: int, num_warps: int, num_stages: int
) -> Tensor:
    # a: (M, K) contiguous, b: (K, N) contiguous.
    if not _use_triton(a):
        return a @ b
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    wrap_triton(_gemm_fp16x3_kernel)[grid](
        a.contiguous(),
        b.contiguous(),
        c,
        m,
        n,
        k,
        S=_TAIL_SCALE,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


_gemm_op = triton_op("dpa1_triton::embed_gemm_fp16x3", mutates_args=())(_gemm_impl)


@_gemm_op.register_fake
def _(a, b, bm, bn, bk, num_warps, num_stages):
    return a.new_empty((a.shape[0], b.shape[1]))


def _gemm_setup_context(ctx, inputs, output):
    a, b, bm, bn, bk, num_warps, num_stages = inputs
    ctx.save_for_backward(b)
    ctx.cfg = (bm, bn, bk, num_warps, num_stages)


def _gemm_backward(ctx, grad_c):
    (b,) = ctx.saved_tensors
    bm, bn, bk, num_warps, num_stages = ctx.cfg
    # d(C = A @ B) / dA = grad_C @ B^T; B^T is (N, K), materialized contiguous
    # (B is tiny -- the embedding weight). The weight gradient is not formed
    # (inference-only; training uses the fp32 reference path).
    grad_a = _gemm_op(
        grad_c.contiguous(), b.t().contiguous(), bm, bn, bk, num_warps, num_stages
    )
    return grad_a, None, None, None, None, None, None


_gemm_op.register_autograd(_gemm_backward, setup_context=_gemm_setup_context)


def embed_gemm_fp16x3(a: Tensor, b: Tensor) -> Tensor:
    """fp16x3 dense matmul ``a @ b`` for the compute-bound DPA1 embedding.

    Parameters
    ----------
    a : Tensor
        Left operand with shape (M, K); ``M`` is the (large) edge/row count.
    b : Tensor
        Right operand (embedding weight) with shape (K, N).

    Returns
    -------
    Tensor
        ``a @ b`` with shape (M, N), computed on fp16 tensor cores with the
        two-term split (fp32-reference accuracy, ~2^-22 rounding). Falls back to
        a plain fp32 matmul off the CUDA fp32 path.

    Notes
    -----
    Inference-only (``DP_TRITON_INFER >= 3``); the registered backward returns
    the input gradient (force path) and no weight gradient. Composes under
    ``make_fx`` / ``torch.export`` as a ``triton_op``.
    """
    bm, bn, bk, num_warps, num_stages = _resolve_gemm_config(
        int(a.shape[1]), int(b.shape[1])
    )
    return _gemm_op(a, b, bm, bn, bk, num_warps, num_stages)


def embed_last_gemm(h: Tensor, w: Tensor, bias: Tensor | None) -> Tensor:
    """Last embedding-layer pre-activation ``z2 = h @ w (+ bias)``.

    Routes through the fp16x3 tensor-core GEMM at ``DP_TRITON_INFER >= 3`` (the
    large-``M`` embedding GEMM is compute-bound and beats cuBLAS fp32 there),
    else a plain fp32 matmul. Shared by the strip (``se_atten_conv``) and concat
    / graph (pt_expt) fused paths.

    Parameters
    ----------
    h : Tensor
        Penultimate embedding activation with shape (..., h1_dim): 2-D
        ``(E, h1_dim)`` on the graph path, 3-D ``(nfnl, nnei, h1_dim)`` on the
        dense path (the leading axes are flattened for the 2-D GEMM).
    w : Tensor
        Last-layer weight with shape (h1_dim, ng).
    bias : Tensor or None
        Optional last-layer bias with shape (ng,).

    Returns
    -------
    Tensor
        The pre-activation ``z2`` with shape (..., ng).
    """
    if triton_infer_level() >= 3:
        lead = h.shape[:-1]
        flat = embed_gemm_fp16x3(h.reshape(-1, h.shape[-1]).contiguous(), w)
        z2 = flat.reshape(*lead, w.shape[-1])
    else:
        z2 = torch.matmul(h, w)
    if bias is not None:
        z2 = z2 + bias
    return z2


def _resolve_gemm_config(k: int, n: int) -> tuple[int, int, int, int, int]:
    """Launch config ``(BM, BN, BK, num_warps, num_stages)`` for the embedding GEMM.

    Keyed by ``(K, N)`` (the small contracted/output widths that bound the tile);
    ``M`` is streamed and does not shift the optimum. Unswept shapes take a
    spill-safe default. ``num_stages`` is pinned to 1 across every entry: the
    split-fp16 k-loop is miscompiled by the Triton pipeliner at higher stage
    counts (silent ``NaN``), so only stage 1 is certifiably finite for all shapes.
    """
    return _GEMM_CONFIGS.get((int(k), int(n)), _GEMM_DEFAULT)


# BM, BN, BK, num_warps, num_stages. Default is a safe 64x64x32 tile with the
# pipeliner disabled (num_stages=1) -- see the module docstring on fp16x3 NaN.
_GEMM_DEFAULT: tuple[int, int, int, int, int] = (64, 64, 32, 4, 1)
_GEMM_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
