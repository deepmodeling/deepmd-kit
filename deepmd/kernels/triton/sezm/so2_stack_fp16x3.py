# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""fp16x3 split-compensated SO(2) mixing-stack operators.

This module provides a tensor-core implementation of the SO(2) mixing stack
that is numerically interchangeable with the fp32 ``sezm_triton::
so2_mixing_stack`` operator while running the block GEMMs on fp16 tensor
cores.  It is selected at ``DP_TRITON_INFER >= 3`` for ``(focus_dim, lmax)``
keys resolved by :func:`~.tile_configs.stack_fp16x3_configs`; all other
shapes keep the fp32 stack.

Numerical scheme
----------------
Each fp32 GEMM ``C = A @ B`` is evaluated as three fp16 tensor-core products
with fp32 accumulation (a two-term Ootomo split)::

    A = A_hi + A_lo,  B = B_hi + B_lo        (fp16 head + fp16 tail)
    C ~= A_hi B_hi + A_hi B_lo + A_lo B_hi   (the A_lo B_lo term, ~2^-22
                                              relative, is dropped)

An fp16 multiply feeding an fp32 accumulator is exact (11 x 11 -> 22-bit
products), so the only error sources are the two split truncations.  The
head product and the two tail corrections accumulate in *separate* fp32
accumulators merged once per tile: chaining all three into one accumulator
absorbs the small tail terms against the large head partial sums each
k-step and doubles the error.  Measured against fp64 on production shapes,
the per-GEMM maximum relative error is indistinguishable from the fp32 FFMA
reference (~5e-7), at roughly 1.6x the FFMA GEMM throughput on H20.

Dynamic-range handling
----------------------
fp16 spans ~[6e-5, 65504] against fp32's ~[1e-38, 3e38]; both ends are
protected with exact power-of-two scalings:

- *Tail underflow.*  The tail of an element below ~1.2e-4 falls out of the
  fp16 subnormal range and the correction silently vanishes (local
  degradation to bare fp16 accuracy).  Tails are therefore stored pre-scaled
  by ``2^11`` (the fp16 mantissa width) and the accumulated correction is
  scaled back in the epilogue; the scaled tail never overflows where the
  head itself does not (``|x_lo * 2^11| <= |x|``).
- *Head overflow.*  The stack input rides the unnormalized residual stream
  (the default SeZM layout applies the equivariant norm after the SO(2)
  update, not before), so the activation operand is pre-scaled by ``2^-4``
  before the split and the merged accumulator is scaled back by ``2^4``.
  Layer inputs measured on production checkpoints peak near 13 with roughly
  6x per-block growth, so the prescale keeps four orders of magnitude of
  headroom below the fp16 maximum for realistic depths.  Weights are static
  and of order one after training and stay unscaled; a checkpoint whose
  stack weights or activations exceed the fp16 head range surfaces loudly
  as NaN on the first evaluation rather than as silent error.

Accuracy contract
-----------------
The scheme perturbs a trained model's outputs at the level of the fp32
rounding itself per GEMM; through a full force evaluation the accumulated
deviation against the fp32 stack is of order 1e-6 on forces (measured
~4e-6 eV/A maximum on a 4096-atom system) and ~1e-7 eV per atom on the
energy.  The rounding step of the scheme is 2^-22 relative -- three orders
of magnitude finer than TF32 -- so the smoothness character of the
potential-energy surface matches fp32.  The level-3 gate exists so this
trade is always an explicit opt-in.

Launch-configuration discipline
-------------------------------
Some ``(num_warps, num_stages)`` combinations of the three-``tl.dot`` k-loop
are miscompiled by the Triton software pipeliner into silent NaN rows at
production edge counts, and the affected set shifts with any change to the
kernel body.  Launch configurations therefore come exclusively from
:func:`~.tile_configs.stack_fp16x3_configs`, whose entries are regenerated
by the fp64-validated sweep (``sweep_tile_configs.py --kernels fp16x3``).
Any edit to a kernel body in this module invalidates every table entry.

Layout and semantics are identical to ``sezm_triton::so2_mixing_stack``
(m-major focus-major rows, raw pre-activations saved for the backward, the
competition weight folded into the final store); the gate, recompute and
pointwise-backward kernels are shared with the fp32 operator.
"""

from __future__ import (
    annotations,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)

from .so2_value_path import (
    SO2_VALUE_PATH_TRITON_AVAILABLE,
    _has_no_edges,
    _mixing_stack_backward_reference,
    _mixing_stack_reference,
    _use_triton,
)
from .tile_configs import (
    GATE_BMM_MIN_FOCUS_DIM,
    gate_config,
    point_config,
    recompute_config,
    stack_fp16x3_configs,
)

__all__ = [
    "STACK_FP16X3_TRITON_AVAILABLE",
    "mixing_stack_fp16x3",
]

STACK_FP16X3_TRITON_AVAILABLE = SO2_VALUE_PATH_TRITON_AVAILABLE

if STACK_FP16X3_TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    from .so2_value_path import (
        _stack_gate_kernel,
        _stack_grad_alpha_kernel,
        _stack_point_bwd_kernel,
        _stack_recompute_kernel,
    )

    @triton.jit
    def _split_fp16_kernel(
        w_ptr,  # (numel,) fp32 weights, contiguous
        hi_ptr,  # (numel,) fp16 head out
        lo_ptr,  # (numel,) fp16 tail out (pre-scaled by 2^11)
        numel,
        BLOCK: tl.constexpr,
    ):
        """Two-term fp16 split evaluated inside Triton.

        The split must not be expressed as aten operations: Inductor's
        codegen keeps pointwise intermediates in fp32 and elides the
        ``fp32 -> fp16 -> fp32`` rounding round-trip, which turns the tail
        into exact zero and silently disables the compensation on the
        compiled path.  A Triton kernel is an opaque leaf for Inductor and
        its ``.to`` conversions round as written.
        """
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        hi = w.to(tl.float16)
        lo = ((w - hi.to(tl.float32)) * 2048.0).to(tl.float16)
        tl.store(hi_ptr + offs, hi, mask=mask)
        tl.store(lo_ptr + offs, lo, mask=mask)

    @triton.jit
    def _dot_fp16x3(a, bh, bl, acc, acc2):
        """fp16x3 compensated dot with prescaled head / scaled-tail terms.

        The activation tile is scaled by ``2^-4`` before the split and its
        tail by a further ``2^11``; the epilogue applies the matching inverse
        factors (``2^4`` on the head accumulator, ``2^4 * 2^-11`` on the tail
        accumulator).
        """
        a_s = a * 0.0625
        a_hi = a_s.to(tl.float16)
        a_lo = ((a_s - a_hi.to(tl.float32)) * 2048.0).to(tl.float16)
        acc = tl.dot(a_hi, bh, acc)
        acc2 = tl.dot(a_hi, bl, acc2)
        acc2 = tl.dot(a_lo, bh, acc2)
        return acc, acc2

    @triton.jit
    def _stack_fp16x3_m0_kernel(
        u_ptr,  # (F, E, ROW) layer input
        wh_ptr,  # (NL, F, M0, M0) fp16 weight head
        wl_ptr,  # (NL, F, M0, M0) fp16 weight tail (pre-scaled by 2^11)
        alpha_ptr,  # (E, F) competition weight (identity epilogue only)
        v_ptr,  # z_all stack (EPILOGUE 0) or the final output (EPILOGUE 1)
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        EPILOGUE: tl.constexpr,  # 0: store raw z; 1: residual (+ alpha) output
        V_EDGE_MAJOR: tl.constexpr,  # v is (E, F, ROW); else focus-major
        APPLY_ALPHA: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``m = 0`` block GEMM ``z = u[:, :M0] @ W0`` (fp16x3 inner product).

        Output strides are derived in-kernel from the layout flag on int64
        offsets: a host-side ``n_edge * ROW`` scalar argument would be
        specialized to int32 by the first (small) compilation and overflow
        on systems beyond ~2^31 / ROW edges.
        """
        M0: tl.constexpr = (L + 1) * CF
        ROW: tl.constexpr = (3 * L + 1) * CF
        NT: tl.constexpr = (M0 + BLOCK_N - 1) // BLOCK_N

        pid = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)
        pid_m = pid // NT
        pid_n = pid % NT

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        u_row = u_ptr + fid * n_edge * ROW + offs_m * ROW
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < M0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = u_row[:, None] + offs_k[None, :]
        w_off = (
            (layer * n_focus + fid) * M0 * M0 + offs_k[:, None] * M0 + offs_n[None, :]
        )
        for _ in range(0, M0, BLOCK_K):
            a = tl.load(a_ptrs, mask=mm, other=0.0)
            bh = tl.load(wh_ptr + w_off, mask=n_mask[None, :], other=0.0)
            bl = tl.load(wl_ptr + w_off, mask=n_mask[None, :], other=0.0)
            acc, acc2 = _dot_fp16x3(a, bh, bl, acc, acc2)
            a_ptrs += BLOCK_K
            w_off += BLOCK_K * M0
        acc = acc * 16.0 + acc2 * 0.0078125  # 2^4 head, 2^4 * 2^-11 tail unscale

        if EPILOGUE == 1:
            u_t = tl.load(
                u_row[:, None] + offs_n[None, :], mask=mm & n_mask[None, :], other=0.0
            )
            acc = acc + u_t
            if APPLY_ALPHA:
                alpha = tl.load(
                    alpha_ptr + offs_m * n_focus + fid, mask=m_mask, other=0.0
                )
                acc = acc * alpha[:, None]
            if V_EDGE_MAJOR:
                v_row = v_ptr + fid * ROW + offs_m * (n_focus * ROW)
            else:
                v_row = v_ptr + fid * n_edge * ROW + offs_m * ROW
            tl.store(v_row[:, None] + offs_n[None, :], acc, mask=mm & n_mask[None, :])
        else:
            z_row = v_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
            tl.store(z_row[:, None] + offs_n[None, :], acc, mask=mm & n_mask[None, :])

    @triton.jit
    def _stack_fp16x3_m1_kernel(
        u_ptr,
        wh_ptr,  # (NL, F, M1, M1) fp16 weight head
        wl_ptr,  # (NL, F, M1, M1) fp16 weight tail (pre-scaled by 2^11)
        sig_ptr,  # (F, E, L*CF) gate sigmoids (HAS_GATE)
        alpha_ptr,
        v_ptr,
        z_ptr,
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        HAS_GATE: tl.constexpr,
        V_EDGE_MAJOR: tl.constexpr,  # v is (E, F, ROW); else focus-major
        APPLY_ALPHA: tl.constexpr,
        SAVE_Z: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``|m| = 1`` block GEMM with the gate / residual / alpha epilogue."""
        M0: tl.constexpr = (L + 1) * CF
        M1: tl.constexpr = 2 * L * CF
        ROW: tl.constexpr = (3 * L + 1) * CF
        LG: tl.constexpr = L * CF
        NT: tl.constexpr = (M1 + BLOCK_N - 1) // BLOCK_N

        pid = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)
        pid_m = pid // NT
        pid_n = pid % NT

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        u_row = u_ptr + fid * n_edge * ROW + offs_m * ROW
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < M1

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = u_row[:, None] + (M0 + offs_k)[None, :]
        w_off = (
            (layer * n_focus + fid) * M1 * M1 + offs_k[:, None] * M1 + offs_n[None, :]
        )
        for _ in range(0, M1, BLOCK_K):
            a = tl.load(a_ptrs, mask=mm, other=0.0)
            bh = tl.load(wh_ptr + w_off, mask=n_mask[None, :], other=0.0)
            bl = tl.load(wl_ptr + w_off, mask=n_mask[None, :], other=0.0)
            acc, acc2 = _dot_fp16x3(a, bh, bl, acc, acc2)
            a_ptrs += BLOCK_K
            w_off += BLOCK_K * M1
        acc = acc * 16.0 + acc2 * 0.0078125  # 2^4 head, 2^4 * 2^-11 tail unscale

        if SAVE_Z:
            z_row = z_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
            tl.store(
                z_row[:, None] + (M0 + offs_n)[None, :], acc, mask=mm & n_mask[None, :]
            )
        if HAS_GATE:
            # Both |m| = 1 stripes of degree group g share gate group g.
            sig_cols = ((offs_n // CF) % L) * CF + (offs_n % CF)
            sig = tl.load(
                sig_ptr + (fid * n_edge + offs_m)[:, None] * LG + sig_cols[None, :],
                mask=mm & n_mask[None, :],
                other=0.0,
            )
            acc = acc * sig
        u_t = tl.load(
            u_row[:, None] + (M0 + offs_n)[None, :],
            mask=mm & n_mask[None, :],
            other=0.0,
        )
        acc = acc + u_t
        if APPLY_ALPHA:
            alpha = tl.load(alpha_ptr + offs_m * n_focus + fid, mask=m_mask, other=0.0)
            acc = acc * alpha[:, None]
        if V_EDGE_MAJOR:
            v_row = v_ptr + fid * ROW + offs_m * (n_focus * ROW)
        else:
            v_row = v_ptr + fid * n_edge * ROW + offs_m * ROW
        tl.store(
            v_row[:, None] + (M0 + offs_n)[None, :], acc, mask=mm & n_mask[None, :]
        )

    @triton.jit
    def _stack_fp16x3_bwd_kernel(
        gz_ptr,  # (F, E, ROW), or the raw upstream gradient when FOLD_ALPHA
        res_ptr,  # (F, E, ROW) residual gradient source; unread if FOLD_ALPHA
        wh_ptr,  # (NL, F, MB, MB) fp16 transposed weight head of this block
        wl_ptr,  # (NL, F, MB, MB) fp16 transposed weight tail (2^11-scaled)
        alpha_ptr,
        gu_ptr,  # (F, E, ROW) layer-input gradient
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        IS_M1: tl.constexpr,  # 0: m = 0 block (offset 0), 1: |m| = 1 block
        G_EDGE_MAJOR: tl.constexpr,  # gz is (E, F, ROW); else focus-major
        FOLD_ALPHA: tl.constexpr,  # gz = g * alpha on the fly; residual == gz
        RES_IS_GZ: tl.constexpr,  # residual equals gz (final layer, no alpha)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Backward GEMM ``g_u = residual + gz @ W^T`` for one ``|m|`` block.

        The two blocks are separate launches (``IS_M1`` 0 / 1) so each
        pipelines with its own swept schedule instead of sharing one
        compromise configuration.
        """
        M0: tl.constexpr = (L + 1) * CF
        MB: tl.constexpr = (2 * L * CF) if IS_M1 else ((L + 1) * CF)
        OFF: tl.constexpr = M0 if IS_M1 else 0
        ROW: tl.constexpr = (3 * L + 1) * CF
        NT: tl.constexpr = (MB + BLOCK_N - 1) // BLOCK_N

        pid = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)
        pid_m = pid // NT
        pid_n = pid % NT

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < MB

        if G_EDGE_MAJOR:
            gz_row = gz_ptr + fid * ROW + offs_m * (n_focus * ROW)
        else:
            gz_row = gz_ptr + fid * n_edge * ROW + offs_m * ROW
        gu_row = gu_ptr + fid * n_edge * ROW + offs_m * ROW
        if FOLD_ALPHA:
            alpha = tl.load(alpha_ptr + offs_m * n_focus + fid, mask=m_mask, other=0.0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = gz_row[:, None] + (OFF + offs_k)[None, :]
        w_off = (
            (layer * n_focus + fid) * MB * MB + offs_k[:, None] * MB + offs_n[None, :]
        )
        for _ in range(0, MB, BLOCK_K):
            a = tl.load(a_ptrs, mask=mm, other=0.0)
            if FOLD_ALPHA:
                a = a * alpha[:, None]
            bh = tl.load(wh_ptr + w_off, mask=n_mask[None, :], other=0.0)
            bl = tl.load(wl_ptr + w_off, mask=n_mask[None, :], other=0.0)
            acc, acc2 = _dot_fp16x3(a, bh, bl, acc, acc2)
            a_ptrs += BLOCK_K
            w_off += BLOCK_K * MB
        acc = acc * 16.0 + acc2 * 0.0078125  # 2^4 head, 2^4 * 2^-11 tail unscale

        col0 = OFF + offs_n
        if FOLD_ALPHA:
            res = tl.load(
                gz_row[:, None] + col0[None, :], mask=mm & n_mask[None, :], other=0.0
            )
            res = res * alpha[:, None]
        elif RES_IS_GZ:
            res = tl.load(
                gz_row[:, None] + col0[None, :], mask=mm & n_mask[None, :], other=0.0
            )
        else:
            res_row = res_ptr + fid * n_edge * ROW + offs_m * ROW
            res = tl.load(
                res_row[:, None] + col0[None, :], mask=mm & n_mask[None, :], other=0.0
            )
        tl.store(gu_row[:, None] + col0[None, :], acc + res, mask=mm & n_mask[None, :])


def _split_fp16(w: Tensor) -> tuple[Tensor, Tensor]:
    """Two-term fp16 split ``w ~= hi + lo * 2^-11`` (contiguous halves).

    The tail is stored pre-scaled by ``2^11`` so it stays representable in
    fp16 over the whole fp32-relevant magnitude range; the kernels apply the
    matching inverse factor in their epilogues (exact powers of two).

    The split runs as a Triton kernel rather than as aten operations: the
    tail is *defined* by an ``fp32 -> fp16 -> fp32`` rounding round-trip, and
    Inductor's pointwise fusion keeps such intermediates in fp32 registers,
    which folds the round-trip away and zeroes the tail on the compiled path.
    """
    w = w.contiguous()
    hi = torch.empty(w.shape, device=w.device, dtype=torch.float16)
    lo = torch.empty(w.shape, device=w.device, dtype=torch.float16)
    numel = w.numel()
    block = 1024
    wrap_triton(_split_fp16_kernel)[(triton.cdiv(numel, block),)](
        w.view(-1), hi.view(-1), lo.view(-1), numel, BLOCK=block
    )
    return hi, lo


def _mixing_stack_fp16x3_impl(
    u0: Tensor,
    alpha: Tensor,
    w0_all: Tensor,
    w1_all: Tensor,
    gw_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(u0):
        return _mixing_stack_reference(
            u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha
        )
    n_focus, n_edge, row = u0.shape
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    configs = stack_fp16x3_configs(focus_dim, lmax)
    if configs is None:
        raise RuntimeError(
            f"no validated fp16x3 configuration for (focus_dim={focus_dim}, "
            f"lmax={lmax}); the caller must route unswept shapes to the fp32 "
            "mixing stack"
        )
    n_gated = gw_all.shape[0]
    z_all = torch.empty(
        (n_gated, n_focus, n_edge, row), device=u0.device, dtype=u0.dtype
    )
    x_local = torch.empty((n_edge, n_focus, row), device=u0.device, dtype=u0.dtype)
    if _has_no_edges(n_edge):
        return x_local, z_all

    # Weight splits are parameter-only and negligible next to the GEMMs.
    w0h, w0l = _split_fp16(w0_all)
    w1h, w1l = _split_fp16(w1_all)

    (bm0, bn0, bk0, w0_warps, w0_stages), (bm1, bn1, bk1, w1_warps, w1_stages) = (
        configs[0],
        configs[1],
    )
    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    gate_bm, gate_w, gate_s = gate_config(focus_dim, lmax)
    sig_by_bmm = focus_dim >= GATE_BMM_MIN_FOCUS_DIM
    sig = torch.empty(
        (n_focus, n_edge, lmax * focus_dim), device=u0.device, dtype=torch.float32
    )
    grid_m0 = (triton.cdiv(n_edge, bm0) * triton.cdiv(m0, bn0), n_focus)
    grid_m1 = (triton.cdiv(n_edge, bm1) * triton.cdiv(m1, bn1), n_focus)

    u = u0
    for layer in range(n_gated):
        out = torch.empty_like(u)
        wrap_triton(_stack_fp16x3_m0_kernel)[grid_m0](
            u,
            w0h,
            w0l,
            u,
            z_all,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            EPILOGUE=0,
            V_EDGE_MAJOR=False,
            APPLY_ALPHA=False,
            BLOCK_M=bm0,
            BLOCK_N=bn0,
            BLOCK_K=bk0,
            num_warps=w0_warps,
            num_stages=w0_stages,
        )
        if sig_by_bmm:
            # Wide-channel regime: sigmoid projection as a cuBLAS bmm on the
            # freshly written l = 0 scalar rows of the pre-activation.
            torch.sigmoid(
                torch.bmm(z_all[layer, :, :, :focus_dim], gw_all[layer]), out=sig
            )
        wrap_triton(_stack_gate_kernel)[(triton.cdiv(n_edge, gate_bm), n_focus)](
            u,
            z_all,
            gw_all,
            out,
            sig,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            SIG_IN=sig_by_bmm,
            BLOCK_M=gate_bm,
            num_warps=gate_w,
            num_stages=gate_s,
        )
        wrap_triton(_stack_fp16x3_m1_kernel)[grid_m1](
            u,
            w1h,
            w1l,
            sig,
            u,
            out,
            z_all,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            HAS_GATE=True,
            V_EDGE_MAJOR=False,
            APPLY_ALPHA=False,
            SAVE_Z=True,
            BLOCK_M=bm1,
            BLOCK_N=bn1,
            BLOCK_K=bk1,
            num_warps=w1_warps,
            num_stages=w1_stages,
        )
        u = out

    # Final identity layer streams straight into the edge-major output layout.
    wrap_triton(_stack_fp16x3_m0_kernel)[grid_m0](
        u,
        w0h,
        w0l,
        alpha,
        x_local,
        n_edge,
        n_gated,
        L=lmax,
        CF=focus_dim,
        EPILOGUE=1,
        V_EDGE_MAJOR=True,
        APPLY_ALPHA=apply_alpha,
        BLOCK_M=bm0,
        BLOCK_N=bn0,
        BLOCK_K=bk0,
        num_warps=w0_warps,
        num_stages=w0_stages,
    )
    wrap_triton(_stack_fp16x3_m1_kernel)[grid_m1](
        u,
        w1h,
        w1l,
        sig,
        alpha,
        x_local,
        u,
        n_edge,
        n_gated,
        L=lmax,
        CF=focus_dim,
        HAS_GATE=False,
        V_EDGE_MAJOR=True,
        APPLY_ALPHA=apply_alpha,
        SAVE_Z=False,
        BLOCK_M=bm1,
        BLOCK_N=bn1,
        BLOCK_K=bk1,
        num_warps=w1_warps,
        num_stages=w1_stages,
    )
    return x_local, z_all


def _mixing_stack_fp16x3_bwd_impl(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(grad_out):
        return _mixing_stack_backward_reference(
            grad_out,
            x_local,
            z_all,
            alpha,
            w0t_all,
            w1t_all,
            gw_all,
            gwt_all,
            lmax,
            focus_dim,
            apply_alpha,
        )
    n_gated, n_focus, n_edge, row = z_all.shape
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    configs = stack_fp16x3_configs(focus_dim, lmax)
    if configs is None:
        raise RuntimeError(
            f"no validated fp16x3 configuration for (focus_dim={focus_dim}, "
            f"lmax={lmax}); the caller must route unswept shapes to the fp32 "
            "mixing stack"
        )
    device, dtype = grad_out.device, grad_out.dtype
    grad_alpha = torch.empty((n_edge, n_focus), device=device, dtype=dtype)
    grad_u0 = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    if _has_no_edges(n_edge):
        return grad_u0, grad_alpha

    w0h, w0l = _split_fp16(w0t_all)
    w1h, w1l = _split_fp16(w1t_all)

    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    (bm0, bn0, bk0, w0_warps, w0_stages), (bm1, bn1, bk1, w1_warps, w1_stages) = (
        configs[2],
        configs[3],
    )
    grid_bwd0 = (triton.cdiv(n_edge, bm0) * triton.cdiv(m0, bn0), n_focus)
    grid_bwd1 = (triton.cdiv(n_edge, bm1) * triton.cdiv(m1, bn1), n_focus)
    point_bm, point_w, point_s = point_config(focus_dim, lmax)

    def launch_bwd_gemms(gz, res, gu, layer, g_edge_major, fold, res_is_gz):
        wrap_triton(_stack_fp16x3_bwd_kernel)[grid_bwd0](
            gz,
            res,
            w0h,
            w0l,
            alpha,
            gu,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            IS_M1=False,
            G_EDGE_MAJOR=g_edge_major,
            FOLD_ALPHA=fold,
            RES_IS_GZ=res_is_gz,
            BLOCK_M=bm0,
            BLOCK_N=bn0,
            BLOCK_K=bk0,
            num_warps=w0_warps,
            num_stages=w0_stages,
        )
        wrap_triton(_stack_fp16x3_bwd_kernel)[grid_bwd1](
            gz,
            res,
            w1h,
            w1l,
            alpha,
            gu,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            IS_M1=True,
            G_EDGE_MAJOR=g_edge_major,
            FOLD_ALPHA=fold,
            RES_IS_GZ=res_is_gz,
            BLOCK_M=bm1,
            BLOCK_N=bn1,
            BLOCK_K=bk1,
            num_warps=w1_warps,
            num_stages=w1_stages,
        )

    # === Final layer: g = gz + gz @ W^T with gz = grad [* alpha] on the fly ===
    g_cur = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    launch_bwd_gemms(grad_out, grad_out, g_cur, n_gated, True, apply_alpha, True)
    if apply_alpha:
        a_bm, a_w, a_s = gate_config(focus_dim, lmax)
        wrap_triton(_stack_grad_alpha_kernel)[(triton.cdiv(n_edge, a_bm), n_focus)](
            grad_out,
            x_local,
            alpha,
            grad_alpha,
            n_edge,
            L=lmax,
            CF=focus_dim,
            BLOCK_M=a_bm,
            num_warps=a_w,
            num_stages=a_s,
        )

    # === Gated layers in reverse; sig / gz buffers are reused across layers ===
    gate_width = lmax * focus_dim
    sig = torch.empty((n_focus, n_edge, gate_width), device=device, dtype=torch.float32)
    gz = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    use_bmm = focus_dim >= GATE_BMM_MIN_FOCUS_DIM
    glogit = (
        torch.empty((n_focus, n_edge, gate_width), device=device, dtype=torch.float32)
        if use_bmm
        else sig
    )
    r_bm, r_w, r_s = recompute_config(focus_dim, lmax)
    for layer in range(n_gated - 1, -1, -1):
        if use_bmm:
            torch.sigmoid(
                torch.bmm(z_all[layer, :, :, :focus_dim], gw_all[layer]), out=sig
            )
        else:
            wrap_triton(_stack_recompute_kernel)[(triton.cdiv(n_edge, r_bm), n_focus)](
                z_all,
                gw_all,
                sig,
                n_edge,
                layer,
                L=lmax,
                CF=focus_dim,
                BLOCK_M=r_bm,
                num_warps=r_w,
                num_stages=r_s,
            )
        wrap_triton(_stack_point_bwd_kernel)[(triton.cdiv(n_edge, point_bm), n_focus)](
            g_cur,
            z_all,
            sig,
            gwt_all,
            gz,
            glogit,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            GLOGIT_OUT=use_bmm,
            BLOCK_M=point_bm,
            num_warps=point_w,
            num_stages=point_s,
        )
        if use_bmm:
            # Gate-logit contraction back to the scalar rows via cuBLAS.
            gz[:, :, :focus_dim] += torch.bmm(glogit, gwt_all[layer])
        g_next = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
        launch_bwd_gemms(gz, g_cur, g_next, layer, False, False, False)
        g_cur = g_next
    return g_cur, grad_alpha


# ======================================================================
# Functional triton_op + fake + autograd registration
# ======================================================================
_mixing_stack_fp16x3_op = torch.library.triton_op(
    "sezm_triton::so2_mixing_stack_fp16x3", mutates_args=()
)(_mixing_stack_fp16x3_impl)
_mixing_stack_fp16x3_bwd_op = torch.library.triton_op(
    "sezm_triton::so2_mixing_stack_fp16x3_bwd", mutates_args=()
)(_mixing_stack_fp16x3_bwd_impl)


@_mixing_stack_fp16x3_op.register_fake
def _(u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha):
    n_focus, n_edge, row = u0.shape
    return (
        u0.new_empty((n_edge, n_focus, row)),
        u0.new_empty((gw_all.shape[0], n_focus, n_edge, row)),
    )


@_mixing_stack_fp16x3_bwd_op.register_fake
def _(
    grad_out,
    x_local,
    z_all,
    alpha,
    w0t_all,
    w1t_all,
    gw_all,
    gwt_all,
    lmax,
    focus_dim,
    apply_alpha,
):
    n_gated, n_focus, n_edge, row = z_all.shape
    return (
        z_all.new_empty((n_focus, n_edge, row)),
        z_all.new_empty((n_edge, n_focus)),
    )


def _setup_context(ctx, inputs, output):
    u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha = inputs
    x_local, z_all = output
    ctx.save_for_backward(alpha, x_local, z_all, w0_all, w1_all, gw_all)
    ctx.lmax = lmax
    ctx.focus_dim = focus_dim
    ctx.apply_alpha = apply_alpha


def _backward(ctx, grad_out, grad_z_unused):
    alpha, x_local, z_all, w0_all, w1_all, gw_all = ctx.saved_tensors
    grad_u0, grad_alpha = _mixing_stack_fp16x3_bwd_op(
        grad_out.contiguous(),
        x_local,
        z_all,
        alpha,
        w0_all.transpose(2, 3).contiguous(),
        w1_all.transpose(2, 3).contiguous(),
        gw_all,
        gw_all.transpose(2, 3).contiguous(),
        ctx.lmax,
        ctx.focus_dim,
        ctx.apply_alpha,
    )
    return (
        grad_u0,
        grad_alpha if ctx.apply_alpha else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_mixing_stack_fp16x3_op.register_autograd(_backward, setup_context=_setup_context)


def mixing_stack_fp16x3(
    u0: Tensor,
    alpha: Tensor,
    w0_all: Tensor,
    w1_all: Tensor,
    gw_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    """Run the SO(2) mixing stack through the fp16x3 tensor-core operator.

    Drop-in replacement for ``sezm_triton::so2_mixing_stack`` on shapes whose
    launch configuration passed the fp64 validation sweep (see the module
    docstring for the numerical scheme and its accuracy contract).

    Parameters
    ----------
    u0 : Tensor
        Focus-major stack input with shape (n_focus, n_edge, row), where
        ``row = (3 * lmax + 1) * focus_dim``.
    alpha : Tensor
        Cross-focus competition weight with shape (n_edge, n_focus).
    w0_all : Tensor
        Stacked ``m = 0`` block weights with shape
        (n_layers, n_focus, M0, M0), (in, out) convention.
    w1_all : Tensor
        Stacked ``|m| = 1`` block weights with shape
        (n_layers, n_focus, M1, M1).
    gw_all : Tensor
        Stacked gate projections with shape
        (n_layers - 1, n_focus, focus_dim, lmax * focus_dim).
    lmax : int
        Maximum spherical harmonic degree.
    focus_dim : int
        Per-focus channel width ``Cf``.
    apply_alpha : bool
        Whether the competition weight is folded into the final store.

    Returns
    -------
    tuple[Tensor, Tensor]
        The edge-major local features with shape (n_edge, n_focus, row) and
        the stacked gated-layer pre-activations with shape
        (n_layers - 1, n_focus, n_edge, row).
    """
    return _mixing_stack_fp16x3_op(
        u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha
    )
