# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""Fused grid pair product for training, on tensor cores.

Every grid operator of the model evaluates
``out = from_grid(to_grid(left) * to_grid(right))`` on coefficient operands
(``GridProduct`` directly, ``GridBranch`` at a single branch through a
softmax over one element). Unfused, the training graph materializes the grid
field -- several times larger than its coefficient operand -- for the
forward, the backward and the force-loss second order, and surrounds each
einsum with full-size layout copies.

The composition is a GEMM-pointwise-GEMM sandwich, structurally the flash
attention pattern with the grid axis in the sequence role: one program owns
one ``(pair, channel-block)`` output tile, walks the grid in blocks, and per
block evaluates the two projection ``tl.dot`` products, the pointwise
product, and the back-projection outer ``tl.dot`` into a resident fp32
accumulator. The grid field never reaches device memory on any
differentiation order, and every contraction runs on the tensor cores --
the register-resident CUDA form of the inference operator evaluates the
same walk at FFMA rate, an order of magnitude below, and loses to the dense
composition on the wide SO(3) shapes this operator serves.

The first order is one kernel (five dots per grid block); the second order
of the force-loss regime is one further kernel: the backward is trilinear
in ``(grad_out, left, right)``, so each curvature term is a traversal with
one operand replaced by its cotangent,

    ggo  = F^T[(T h_gl) (T r)] + F^T[(T l) (T h_gr)]
    g2_l = T^T[(F go) (T h_gr)]
    g2_r = T^T[(F go) (T h_gl)]

and all three share one walk (five projection dots, three outer dots). The
projectors are fixed quadrature matrices, so no parameter gradient exists.

Numerics follow the ambient dtype exactly as the dense einsum composition
does under autocast: the operators carry a CUDA autocast rule, so AMP hands
the kernels bf16 operands with fp32 accumulators (the tensor-core
contract); fp32 operands run IEEE fp32 dots (TF32 disabled), matching the
non-autocast dense path.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)

__all__ = [
    "GRID_PAIR_TRITON_AVAILABLE",
    "grid_pair_train",
]

try:
    import triton
    import triton.language as tl

    GRID_PAIR_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton ships with torch cuda builds
    GRID_PAIR_TRITON_AVAILABLE = False


if GRID_PAIR_TRITON_AVAILABLE:

    @triton.jit
    def _coeff_offsets(
        pair,
        slot,
        channel,
        stride_batch: tl.constexpr,
        stride_coeff: tl.constexpr,
        stride_focus: tl.constexpr,
        stride_channel: tl.constexpr,
        PACKED: tl.constexpr,
        N_FOCUS: tl.constexpr,
        N_FRAMES: tl.constexpr,
        C_ALL: tl.constexpr,
    ):
        """Map a packed ``(pair, slot, channel)`` tile onto an NDFC tensor."""
        if PACKED:
            return (
                pair * stride_batch
                + slot[:, None] * stride_coeff
                + channel[None, :] * stride_channel
            )
        batch = pair // N_FOCUS
        focus = pair % N_FOCUS
        degree = slot // N_FRAMES
        frame = slot % N_FRAMES
        packed_channel = frame[:, None] * C_ALL + channel[None, :]
        return (
            batch * stride_batch
            + degree[:, None] * stride_coeff
            + focus * stride_focus
            + packed_channel * stride_channel
        )

    @triton.jit
    def _grid_pair_fwd_kernel(
        left_ptr,
        right_ptr,
        tg_ptr,
        fg_ptr,
        out_ptr,
        left_s0: tl.constexpr,
        left_s1: tl.constexpr,
        left_s2: tl.constexpr,
        left_s3: tl.constexpr,
        right_s0: tl.constexpr,
        right_s1: tl.constexpr,
        right_s2: tl.constexpr,
        right_s3: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        n_pair,
        n_grid,
        PACKED: tl.constexpr,
        N_FOCUS: tl.constexpr,
        N_FRAMES: tl.constexpr,
        P_DIM: tl.constexpr,
        P_HI: tl.constexpr,
        P_LO: tl.constexpr,
        C_ALL: tl.constexpr,
        C_BLK: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Evaluate ``out = F^T[(T l) (T r)]``, one output tile per program.

        The slot axis is covered by a power-of-two high segment and an
        optional low segment (``P_LO == 0`` compiles it away), so a slot
        count just past a power of two pads to the next 32 instead of the
        next power of two.
        """
        pair = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        c_idx = cb * C_BLK + tl.arange(0, C_BLK)
        c_mask = c_idx < C_ALL

        p_hi = tl.arange(0, P_HI)
        hi_mask = p_hi < P_DIM
        left_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            left_s0,
            left_s1,
            left_s2,
            left_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        right_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            right_s0,
            right_s1,
            right_s2,
            right_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        out_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            out_s0,
            out_s1,
            out_s2,
            out_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        m_hi = hi_mask[:, None] & c_mask[None, :]
        lv_hi = tl.load(left_ptr + left_hi, mask=m_hi, other=0.0)
        rv_hi = tl.load(right_ptr + right_hi, mask=m_hi, other=0.0)
        acc_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        if P_LO > 0:
            p_lo = P_HI + tl.arange(0, P_LO)
            lo_mask = p_lo < P_DIM
            left_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                left_s0,
                left_s1,
                left_s2,
                left_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            right_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                right_s0,
                right_s1,
                right_s2,
                right_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            out_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                out_s0,
                out_s1,
                out_s2,
                out_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            m_lo = lo_mask[:, None] & c_mask[None, :]
            lv_lo = tl.load(left_ptr + left_lo, mask=m_lo, other=0.0)
            rv_lo = tl.load(right_ptr + right_lo, mask=m_lo, other=0.0)
            acc_lo = tl.zeros((P_LO, C_BLK), dtype=tl.float32)

        for g0 in range(0, n_grid, BLOCK_G):
            g_idx = g0 + tl.arange(0, BLOCK_G)
            g_mask = g_idx < n_grid
            prj_hi = g_idx[:, None] * P_DIM + p_hi[None, :]
            pm_hi = g_mask[:, None] & hi_mask[None, :]
            tg_hi = tl.load(tg_ptr + prj_hi, mask=pm_hi, other=0.0)
            lg = tl.dot(tg_hi, lv_hi, allow_tf32=ALLOW_TF32)
            rg = tl.dot(tg_hi, rv_hi, allow_tf32=ALLOW_TF32)
            if P_LO > 0:
                prj_lo = g_idx[:, None] * P_DIM + p_lo[None, :]
                pm_lo = g_mask[:, None] & lo_mask[None, :]
                tg_lo = tl.load(tg_ptr + prj_lo, mask=pm_lo, other=0.0)
                lg += tl.dot(tg_lo, lv_lo, allow_tf32=ALLOW_TF32)
                rg += tl.dot(tg_lo, rv_lo, allow_tf32=ALLOW_TF32)
            fg_hi = tl.load(fg_ptr + prj_hi, mask=pm_hi, other=0.0)
            prod = (lg * rg).to(fg_hi.dtype)
            acc_hi += tl.dot(tl.trans(fg_hi), prod, allow_tf32=ALLOW_TF32)
            if P_LO > 0:
                fg_lo = tl.load(fg_ptr + prj_lo, mask=pm_lo, other=0.0)
                acc_lo += tl.dot(tl.trans(fg_lo), prod, allow_tf32=ALLOW_TF32)

        tl.store(out_ptr + out_hi, acc_hi.to(out_ptr.dtype.element_ty), mask=m_hi)
        if P_LO > 0:
            tl.store(out_ptr + out_lo, acc_lo.to(out_ptr.dtype.element_ty), mask=m_lo)

    @triton.jit
    def _grid_pair_bwd_kernel(
        go_ptr,
        left_ptr,
        right_ptr,
        tg_ptr,
        fg_ptr,
        gl_ptr,
        gr_ptr,
        go_s0: tl.constexpr,
        go_s1: tl.constexpr,
        go_s2: tl.constexpr,
        go_s3: tl.constexpr,
        left_s0: tl.constexpr,
        left_s1: tl.constexpr,
        left_s2: tl.constexpr,
        left_s3: tl.constexpr,
        right_s0: tl.constexpr,
        right_s1: tl.constexpr,
        right_s2: tl.constexpr,
        right_s3: tl.constexpr,
        gl_s0: tl.constexpr,
        gl_s1: tl.constexpr,
        gl_s2: tl.constexpr,
        gl_s3: tl.constexpr,
        gr_s0: tl.constexpr,
        gr_s1: tl.constexpr,
        gr_s2: tl.constexpr,
        gr_s3: tl.constexpr,
        n_pair,
        n_grid,
        PACKED: tl.constexpr,
        N_FOCUS: tl.constexpr,
        N_FRAMES: tl.constexpr,
        P_DIM: tl.constexpr,
        P_HI: tl.constexpr,
        P_LO: tl.constexpr,
        C_ALL: tl.constexpr,
        C_BLK: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """g_l = T^T[(F go)(T r)], g_r = T^T[(F go)(T l)], one shared walk."""
        pair = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        c_idx = cb * C_BLK + tl.arange(0, C_BLK)
        c_mask = c_idx < C_ALL

        p_hi = tl.arange(0, P_HI)
        hi_mask = p_hi < P_DIM
        go_hi_offset = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            go_s0,
            go_s1,
            go_s2,
            go_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        left_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            left_s0,
            left_s1,
            left_s2,
            left_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        right_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            right_s0,
            right_s1,
            right_s2,
            right_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        gl_hi_offset = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            gl_s0,
            gl_s1,
            gl_s2,
            gl_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        gr_hi_offset = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            gr_s0,
            gr_s1,
            gr_s2,
            gr_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        m_hi = hi_mask[:, None] & c_mask[None, :]
        lv_hi = tl.load(left_ptr + left_hi, mask=m_hi, other=0.0)
        rv_hi = tl.load(right_ptr + right_hi, mask=m_hi, other=0.0)
        go_hi = tl.load(go_ptr + go_hi_offset, mask=m_hi, other=0.0)
        gl_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        gr_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        if P_LO > 0:
            p_lo = P_HI + tl.arange(0, P_LO)
            lo_mask = p_lo < P_DIM
            go_lo_offset = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                go_s0,
                go_s1,
                go_s2,
                go_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            left_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                left_s0,
                left_s1,
                left_s2,
                left_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            right_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                right_s0,
                right_s1,
                right_s2,
                right_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            gl_lo_offset = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                gl_s0,
                gl_s1,
                gl_s2,
                gl_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            gr_lo_offset = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                gr_s0,
                gr_s1,
                gr_s2,
                gr_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            m_lo = lo_mask[:, None] & c_mask[None, :]
            lv_lo = tl.load(left_ptr + left_lo, mask=m_lo, other=0.0)
            rv_lo = tl.load(right_ptr + right_lo, mask=m_lo, other=0.0)
            go_lo = tl.load(go_ptr + go_lo_offset, mask=m_lo, other=0.0)
            gl_lo = tl.zeros((P_LO, C_BLK), dtype=tl.float32)
            gr_lo = tl.zeros((P_LO, C_BLK), dtype=tl.float32)

        for g0 in range(0, n_grid, BLOCK_G):
            g_idx = g0 + tl.arange(0, BLOCK_G)
            g_mask = g_idx < n_grid
            prj_hi = g_idx[:, None] * P_DIM + p_hi[None, :]
            pm_hi = g_mask[:, None] & hi_mask[None, :]
            tg_hi = tl.load(tg_ptr + prj_hi, mask=pm_hi, other=0.0)
            fg_hi = tl.load(fg_ptr + prj_hi, mask=pm_hi, other=0.0)
            lg = tl.dot(tg_hi, lv_hi, allow_tf32=ALLOW_TF32)
            rg = tl.dot(tg_hi, rv_hi, allow_tf32=ALLOW_TF32)
            gv = tl.dot(fg_hi, go_hi, allow_tf32=ALLOW_TF32)
            if P_LO > 0:
                prj_lo = g_idx[:, None] * P_DIM + p_lo[None, :]
                pm_lo = g_mask[:, None] & lo_mask[None, :]
                tg_lo = tl.load(tg_ptr + prj_lo, mask=pm_lo, other=0.0)
                fg_lo = tl.load(fg_ptr + prj_lo, mask=pm_lo, other=0.0)
                lg += tl.dot(tg_lo, lv_lo, allow_tf32=ALLOW_TF32)
                rg += tl.dot(tg_lo, rv_lo, allow_tf32=ALLOW_TF32)
                gv += tl.dot(fg_lo, go_lo, allow_tf32=ALLOW_TF32)
            wl = (gv * rg).to(tg_hi.dtype)
            wr = (gv * lg).to(tg_hi.dtype)
            tgt_hi = tl.trans(tg_hi)
            gl_hi += tl.dot(tgt_hi, wl, allow_tf32=ALLOW_TF32)
            gr_hi += tl.dot(tgt_hi, wr, allow_tf32=ALLOW_TF32)
            if P_LO > 0:
                tgt_lo = tl.trans(tg_lo)
                gl_lo += tl.dot(tgt_lo, wl, allow_tf32=ALLOW_TF32)
                gr_lo += tl.dot(tgt_lo, wr, allow_tf32=ALLOW_TF32)

        tl.store(gl_ptr + gl_hi_offset, gl_hi.to(gl_ptr.dtype.element_ty), mask=m_hi)
        tl.store(gr_ptr + gr_hi_offset, gr_hi.to(gr_ptr.dtype.element_ty), mask=m_hi)
        if P_LO > 0:
            tl.store(
                gl_ptr + gl_lo_offset, gl_lo.to(gl_ptr.dtype.element_ty), mask=m_lo
            )
            tl.store(
                gr_ptr + gr_lo_offset, gr_lo.to(gr_ptr.dtype.element_ty), mask=m_lo
            )

    @triton.jit
    def _grid_pair_bwd2_kernel(
        hgl_ptr,
        hgr_ptr,
        go_ptr,
        left_ptr,
        right_ptr,
        tg_ptr,
        fg_ptr,
        ggo_ptr,
        g2l_ptr,
        g2r_ptr,
        hgl_s0: tl.constexpr,
        hgl_s1: tl.constexpr,
        hgl_s2: tl.constexpr,
        hgl_s3: tl.constexpr,
        hgr_s0: tl.constexpr,
        hgr_s1: tl.constexpr,
        hgr_s2: tl.constexpr,
        hgr_s3: tl.constexpr,
        go_s0: tl.constexpr,
        go_s1: tl.constexpr,
        go_s2: tl.constexpr,
        go_s3: tl.constexpr,
        left_s0: tl.constexpr,
        left_s1: tl.constexpr,
        left_s2: tl.constexpr,
        left_s3: tl.constexpr,
        right_s0: tl.constexpr,
        right_s1: tl.constexpr,
        right_s2: tl.constexpr,
        right_s3: tl.constexpr,
        ggo_s0: tl.constexpr,
        ggo_s1: tl.constexpr,
        ggo_s2: tl.constexpr,
        ggo_s3: tl.constexpr,
        g2l_s0: tl.constexpr,
        g2l_s1: tl.constexpr,
        g2l_s2: tl.constexpr,
        g2l_s3: tl.constexpr,
        g2r_s0: tl.constexpr,
        g2r_s1: tl.constexpr,
        g2r_s2: tl.constexpr,
        g2r_s3: tl.constexpr,
        n_pair,
        n_grid,
        PACKED: tl.constexpr,
        N_FOCUS: tl.constexpr,
        N_FRAMES: tl.constexpr,
        P_DIM: tl.constexpr,
        P_HI: tl.constexpr,
        P_LO: tl.constexpr,
        C_ALL: tl.constexpr,
        C_BLK: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Force-regime curvature: three outputs off one grid walk."""
        pair = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        c_idx = cb * C_BLK + tl.arange(0, C_BLK)
        c_mask = c_idx < C_ALL

        p_hi = tl.arange(0, P_HI)
        hi_mask = p_hi < P_DIM
        hgl_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            hgl_s0,
            hgl_s1,
            hgl_s2,
            hgl_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        hgr_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            hgr_s0,
            hgr_s1,
            hgr_s2,
            hgr_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        go_hi_offset = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            go_s0,
            go_s1,
            go_s2,
            go_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        left_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            left_s0,
            left_s1,
            left_s2,
            left_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        right_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            right_s0,
            right_s1,
            right_s2,
            right_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        ggo_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            ggo_s0,
            ggo_s1,
            ggo_s2,
            ggo_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        g2l_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            g2l_s0,
            g2l_s1,
            g2l_s2,
            g2l_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        g2r_hi = _coeff_offsets(
            pair,
            p_hi,
            c_idx,
            g2r_s0,
            g2r_s1,
            g2r_s2,
            g2r_s3,
            PACKED,
            N_FOCUS,
            N_FRAMES,
            C_ALL,
        )
        m_hi = hi_mask[:, None] & c_mask[None, :]
        lv_hi = tl.load(left_ptr + left_hi, mask=m_hi, other=0.0)
        rv_hi = tl.load(right_ptr + right_hi, mask=m_hi, other=0.0)
        go_hi = tl.load(go_ptr + go_hi_offset, mask=m_hi, other=0.0)
        hl_hi = tl.load(hgl_ptr + hgl_hi, mask=m_hi, other=0.0)
        hr_hi = tl.load(hgr_ptr + hgr_hi, mask=m_hi, other=0.0)
        ao_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        al_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        ar_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        if P_LO > 0:
            p_lo = P_HI + tl.arange(0, P_LO)
            lo_mask = p_lo < P_DIM
            hgl_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                hgl_s0,
                hgl_s1,
                hgl_s2,
                hgl_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            hgr_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                hgr_s0,
                hgr_s1,
                hgr_s2,
                hgr_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            go_lo_offset = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                go_s0,
                go_s1,
                go_s2,
                go_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            left_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                left_s0,
                left_s1,
                left_s2,
                left_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            right_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                right_s0,
                right_s1,
                right_s2,
                right_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            ggo_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                ggo_s0,
                ggo_s1,
                ggo_s2,
                ggo_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            g2l_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                g2l_s0,
                g2l_s1,
                g2l_s2,
                g2l_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            g2r_lo = _coeff_offsets(
                pair,
                p_lo,
                c_idx,
                g2r_s0,
                g2r_s1,
                g2r_s2,
                g2r_s3,
                PACKED,
                N_FOCUS,
                N_FRAMES,
                C_ALL,
            )
            m_lo = lo_mask[:, None] & c_mask[None, :]
            lv_lo = tl.load(left_ptr + left_lo, mask=m_lo, other=0.0)
            rv_lo = tl.load(right_ptr + right_lo, mask=m_lo, other=0.0)
            go_lo = tl.load(go_ptr + go_lo_offset, mask=m_lo, other=0.0)
            hl_lo = tl.load(hgl_ptr + hgl_lo, mask=m_lo, other=0.0)
            hr_lo = tl.load(hgr_ptr + hgr_lo, mask=m_lo, other=0.0)
            ao_lo = tl.zeros((P_LO, C_BLK), dtype=tl.float32)
            al_lo = tl.zeros((P_LO, C_BLK), dtype=tl.float32)
            ar_lo = tl.zeros((P_LO, C_BLK), dtype=tl.float32)

        for g0 in range(0, n_grid, BLOCK_G):
            g_idx = g0 + tl.arange(0, BLOCK_G)
            g_mask = g_idx < n_grid
            prj_hi = g_idx[:, None] * P_DIM + p_hi[None, :]
            pm_hi = g_mask[:, None] & hi_mask[None, :]
            tg_hi = tl.load(tg_ptr + prj_hi, mask=pm_hi, other=0.0)
            fg_hi = tl.load(fg_ptr + prj_hi, mask=pm_hi, other=0.0)
            lg = tl.dot(tg_hi, lv_hi, allow_tf32=ALLOW_TF32)
            rg = tl.dot(tg_hi, rv_hi, allow_tf32=ALLOW_TF32)
            gv = tl.dot(fg_hi, go_hi, allow_tf32=ALLOW_TF32)
            hlg = tl.dot(tg_hi, hl_hi, allow_tf32=ALLOW_TF32)
            hrg = tl.dot(tg_hi, hr_hi, allow_tf32=ALLOW_TF32)
            if P_LO > 0:
                prj_lo = g_idx[:, None] * P_DIM + p_lo[None, :]
                pm_lo = g_mask[:, None] & lo_mask[None, :]
                tg_lo = tl.load(tg_ptr + prj_lo, mask=pm_lo, other=0.0)
                fg_lo = tl.load(fg_ptr + prj_lo, mask=pm_lo, other=0.0)
                lg += tl.dot(tg_lo, lv_lo, allow_tf32=ALLOW_TF32)
                rg += tl.dot(tg_lo, rv_lo, allow_tf32=ALLOW_TF32)
                gv += tl.dot(fg_lo, go_lo, allow_tf32=ALLOW_TF32)
                hlg += tl.dot(tg_lo, hl_lo, allow_tf32=ALLOW_TF32)
                hrg += tl.dot(tg_lo, hr_lo, allow_tf32=ALLOW_TF32)
            wo = (hlg * rg + lg * hrg).to(fg_hi.dtype)
            wl = (gv * hrg).to(tg_hi.dtype)
            wr = (gv * hlg).to(tg_hi.dtype)
            tgt_hi = tl.trans(tg_hi)
            ao_hi += tl.dot(tl.trans(fg_hi), wo, allow_tf32=ALLOW_TF32)
            al_hi += tl.dot(tgt_hi, wl, allow_tf32=ALLOW_TF32)
            ar_hi += tl.dot(tgt_hi, wr, allow_tf32=ALLOW_TF32)
            if P_LO > 0:
                tgt_lo = tl.trans(tg_lo)
                ao_lo += tl.dot(tl.trans(fg_lo), wo, allow_tf32=ALLOW_TF32)
                al_lo += tl.dot(tgt_lo, wl, allow_tf32=ALLOW_TF32)
                ar_lo += tl.dot(tgt_lo, wr, allow_tf32=ALLOW_TF32)

        tl.store(ggo_ptr + ggo_hi, ao_hi.to(ggo_ptr.dtype.element_ty), mask=m_hi)
        tl.store(g2l_ptr + g2l_hi, al_hi.to(g2l_ptr.dtype.element_ty), mask=m_hi)
        tl.store(g2r_ptr + g2r_hi, ar_hi.to(g2r_ptr.dtype.element_ty), mask=m_hi)
        if P_LO > 0:
            tl.store(ggo_ptr + ggo_lo, ao_lo.to(ggo_ptr.dtype.element_ty), mask=m_lo)
            tl.store(g2l_ptr + g2l_lo, al_lo.to(g2l_ptr.dtype.element_ty), mask=m_lo)
            tl.store(g2r_ptr + g2r_lo, ar_lo.to(g2r_ptr.dtype.element_ty), mask=m_lo)


def _next_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _pack(value: Tensor, n_frames: int) -> tuple[Tensor, tuple[int, ...]]:
    """Reorder ``(N, D, F, K*C)`` to the compact ``(N*F, P, C)`` layout.

    The focus axis strides between the degree and frame axes of the logical
    slot ``p = (d, k)``; one contiguous copy at coefficient resolution is a
    small fraction of the grid-field traffic the kernels avoid.
    """
    n_batch, coeff_dim, n_focus, kc = value.shape
    c_per = kc // n_frames
    packed = (
        value.reshape(n_batch, coeff_dim, n_focus, n_frames, c_per)
        .permute(0, 2, 1, 3, 4)
        .reshape(n_batch * n_focus, coeff_dim * n_frames, c_per)
        .contiguous()
    )
    return packed, (n_batch, coeff_dim, n_focus, kc)


def _unpack(value: Tensor, shape: tuple[int, ...], n_frames: int) -> Tensor:
    # The permute back to the frame-packed layout materializes: the operator
    # contract (and the fake tensors the compile pipeline reasons with)
    # promises contiguous outputs.
    n_batch, coeff_dim, n_focus, kc = shape
    return (
        value.reshape(n_batch, n_focus, coeff_dim, n_frames, kc // n_frames)
        .permute(0, 2, 1, 3, 4)
        .reshape(n_batch, coeff_dim, n_focus, kc)
        .contiguous()
    )


_LAUNCH_CACHE: dict[tuple, tuple[int, int, int]] = {}

# Exact production-shape winners on RTX PRO 6000 Blackwell.  The grid-pair
# kernels hold one, two or three ``(P_PAD, C_BLK)`` accumulator tiles, so the
# best channel width and grid tile change independently across differentiation
# orders.  Other devices and uncovered shapes retain the spill-safe launch
# search below.
_BLACKWELL_LAUNCH_CONFIGS = {
    # (kernel, P, C, F, frames, packed, grid, dtype) -> (C_BLK, BLOCK_G, stages)
    (
        "_grid_pair_bwd_kernel",
        12,
        32,
        1,
        1,
        True,
        24,
        torch.bfloat16,
    ): (32, 128, 2),
    (
        "_grid_pair_bwd2_kernel",
        27,
        32,
        1,
        1,
        True,
        104,
        torch.bfloat16,
    ): (32, 32, 1),
    (
        "_grid_pair_bwd_kernel",
        75,
        64,
        1,
        1,
        True,
        296,
        torch.bfloat16,
    ): (32, 16, 1),
    (
        "_grid_pair_bwd2_kernel",
        75,
        64,
        1,
        1,
        True,
        296,
        torch.bfloat16,
    ): (16, 64, 2),
    (
        "_grid_pair_fwd_kernel",
        108,
        64,
        2,
        3,
        False,
        344,
        torch.bfloat16,
    ): (32, 64, 1),
    (
        "_grid_pair_bwd_kernel",
        108,
        64,
        2,
        3,
        False,
        344,
        torch.bfloat16,
    ): (32, 64, 1),
    (
        "_grid_pair_bwd_kernel",
        147,
        96,
        2,
        3,
        False,
        584,
        torch.bfloat16,
    ): (64, 32, 1),
    (
        "_grid_pair_bwd2_kernel",
        147,
        96,
        2,
        3,
        False,
        584,
        torch.bfloat16,
    ): (16, 32, 2),
    (
        "_grid_pair_fwd_kernel",
        147,
        256,
        1,
        1,
        True,
        584,
        torch.bfloat16,
    ): (64, 64, 1),
    (
        "_grid_pair_bwd_kernel",
        147,
        256,
        1,
        1,
        True,
        584,
        torch.bfloat16,
    ): (64, 32, 1),
    (
        "_grid_pair_bwd2_kernel",
        147,
        256,
        1,
        1,
        True,
        584,
        torch.bfloat16,
    ): (16, 32, 1),
}


def _built_in_launch_config(
    device_name: str, shape_key: tuple
) -> tuple[int, int, int] | None:
    """Return the built-in launch for one exact grid-pair shape."""
    if device_name.startswith("NVIDIA RTX PRO 6000 Blackwell"):
        return _BLACKWELL_LAUNCH_CONFIGS.get(shape_key)
    return None


def _launch(
    kernel,
    value: Tensor,
    n_grid: int,
    n_frames: int,
    packed: bool,
    args: tuple,
    n_acc: int,
) -> None:
    """Launch with the largest tile the register and shared budgets admit.

    The channel block is capped so the ``n_acc`` fp32 accumulator tiles
    ``(P_PAD, C_BLK)`` stay register-resident (a spilled accumulator is
    read-modify-written through local memory once per grid block, which
    dominated the three-accumulator second-order kernel before the cap).
    The exact shared footprint additionally depends on Triton's internal
    staging (dot operand buffers, transpose scratch), so candidates are
    tried from the most to the least aggressive and the first that compiles
    is cached per device and exact operator shape.  Swept built-in launches
    take precedence where available and fall back to the same compile search
    if a later Triton version rejects one.
    """
    n_batch, coeff_dim, n_focus, packed_channels = value.shape
    n_pair = n_batch * n_focus
    p_dim = coeff_dim * n_frames
    c_per = packed_channels // n_frames
    # The slot axis is covered by the largest power of two below the count
    # plus an optional low segment for the remainder, so 147 (degree six)
    # pads to 128 + 32 and 75 (degree four) to 64 + 16 instead of the next
    # power of two. The high segment keeps the tensor-core minimum of 16.
    p_hi = max(16, 1 << (p_dim.bit_length() - 1))
    p_lo = _next_pow2(p_dim - p_hi) if p_dim > p_hi else 0
    p_eff = p_hi + p_lo
    c_top = min(64, _next_pow2(c_per), max(16, _next_pow2(4096 // (n_acc * p_eff))))
    shape_key = (
        kernel.fn.__name__,
        p_dim,
        c_per,
        n_focus,
        n_frames,
        packed,
        n_grid,
        value.dtype,
    )
    device_name = torch.cuda.get_device_name(value.device)
    key = (device_name, *shape_key)
    generic_candidates = [
        (c_blk, block_g, stages)
        for c_blk in (c_top, 32, 16)
        if c_blk <= c_top
        for block_g, stages in ((64, 2), (32, 2), (32, 1), (16, 1))
    ]
    if key in _LAUNCH_CACHE:
        candidates = [_LAUNCH_CACHE[key]]
    else:
        built_in = _built_in_launch_config(device_name, shape_key)
        candidates = (
            generic_candidates
            if built_in is None
            else [built_in, *(cfg for cfg in generic_candidates if cfg != built_in)]
        )
    for c_blk, block_g, stages in candidates:
        grid = (n_pair, (c_per + c_blk - 1) // c_blk)
        try:
            wrap_triton(kernel)[grid](
                *args,
                n_pair,
                n_grid=n_grid,
                PACKED=packed,
                N_FOCUS=n_focus,
                N_FRAMES=n_frames,
                P_DIM=p_dim,
                P_HI=p_hi,
                P_LO=p_lo,
                C_ALL=c_per,
                C_BLK=c_blk,
                ALLOW_TF32=False,
                BLOCK_G=block_g,
                num_warps=8 if c_blk >= 64 else 4,
                num_stages=stages,
            )
        except triton.runtime.errors.OutOfResources:
            continue
        _LAUNCH_CACHE[key] = (c_blk, block_g, stages)
        return
    raise _NoViableConfig(p_dim, c_per)


def _strides(*values: Tensor) -> tuple[int, ...]:
    """Flatten the logical NDFC strides of a kernel's tensor operands."""
    return tuple(int(stride) for value in values for stride in value.stride())


def _kernel_layout(
    values: tuple[Tensor, ...],
    n_frames: int,
) -> tuple[tuple[Tensor, ...], int, tuple[int, ...] | None]:
    """Select the coefficient layout used by the Triton kernels.

    A single focus has no intervening focus axis, so packing only collapses
    adjacent dimensions and the kernels retain linear coefficient addressing.
    Multiple focuses require a materializing permutation; those shapes stay in
    the native layout so the kernels consume the producer strides directly.
    """
    shape = tuple(int(size) for size in values[0].shape)
    if shape[2] != 1:
        return values, n_frames, None
    packed = tuple(_pack(value, n_frames)[0].unsqueeze(2) for value in values)
    return packed, 1, shape


def _restore_layout(
    value: Tensor,
    shape: tuple[int, ...] | None,
    n_frames: int,
) -> Tensor:
    """Restore a single-focus packed result to the operator contract."""
    if shape is None:
        return value
    return _unpack(value.squeeze(2), shape, n_frames)


class _NoViableConfig(Exception):
    """No launch configuration fits the shared-memory budget."""

    def __init__(self, p_dim: int, c_per: int) -> None:
        super().__init__(f"P={p_dim}, C={c_per}")


def _eager_packed(
    op: str,
    to_grid: Tensor,
    from_grid: Tensor,
    *tensors: Tensor,
) -> tuple[Tensor, ...]:
    """Eager einsum fallback on the packed layout.

    Serves the shapes whose padded tiles exceed the shared-memory budget --
    the wide-slot fp32 regime of the exact-precision harnesses; the
    production bf16 shapes never reach it.
    """
    tg = to_grid
    fg = from_grid
    if op == "fwd":
        (lp, rp) = tensors
        return (
            torch.einsum(
                "gp,ngc->npc",
                fg,
                torch.einsum("gp,npc->ngc", tg, lp)
                * torch.einsum("gp,npc->ngc", tg, rp),
            ),
        )
    if op == "bwd":
        (gp, lp, rp) = tensors
        gv = torch.einsum("gp,npc->ngc", fg, gp)
        lg = torch.einsum("gp,npc->ngc", tg, lp)
        rg = torch.einsum("gp,npc->ngc", tg, rp)
        return (
            torch.einsum("gp,ngc->npc", tg, gv * rg),
            torch.einsum("gp,ngc->npc", tg, gv * lg),
        )
    (hlp, hrp, gp, lp, rp) = tensors
    gv = torch.einsum("gp,npc->ngc", fg, gp)
    lg = torch.einsum("gp,npc->ngc", tg, lp)
    rg = torch.einsum("gp,npc->ngc", tg, rp)
    hlg = torch.einsum("gp,npc->ngc", tg, hlp)
    hrg = torch.einsum("gp,npc->ngc", tg, hrp)
    return (
        torch.einsum("gp,ngc->npc", fg, hlg * rg + lg * hrg),
        torch.einsum("gp,ngc->npc", tg, gv * hrg),
        torch.einsum("gp,ngc->npc", tg, gv * hlg),
    )


def _train_impl(
    left: Tensor,
    right: Tensor,
    to_grid: Tensor,
    from_grid: Tensor,
    n_frames: int,
) -> Tensor:
    tg = to_grid.contiguous()
    fg = from_grid.contiguous()
    (kernel_values, kernel_frames, shape) = _kernel_layout((left, right), n_frames)
    kernel_left, kernel_right = kernel_values
    out = torch.empty_like(kernel_left, memory_format=torch.contiguous_format)
    try:
        _launch(
            _grid_pair_fwd_kernel,
            kernel_left,
            int(tg.shape[0]),
            kernel_frames,
            shape is not None,
            (
                kernel_left,
                kernel_right,
                tg,
                fg,
                out,
                *_strides(kernel_left, kernel_right, out),
            ),
            1,
        )
    except _NoViableConfig:
        lp, kernel_shape = _pack(kernel_left, kernel_frames)
        rp, _ = _pack(kernel_right, kernel_frames)
        (out,) = _eager_packed("fwd", tg, fg, lp, rp)
        out = _unpack(out, kernel_shape, kernel_frames)
    return _restore_layout(out, shape, n_frames)


_train_op = torch.library.triton_op(
    "sezm_triton::grid_pair_train",
    _train_impl,
    mutates_args=(),
)


@_train_op.register_fake
def _(left, right, to_grid, from_grid, n_frames):
    del right, to_grid, from_grid, n_frames
    return left.new_empty(left.shape)


def _train_bwd_impl(
    grad_out: Tensor,
    left: Tensor,
    right: Tensor,
    to_grid: Tensor,
    from_grid: Tensor,
    n_frames: int,
) -> tuple[Tensor, Tensor]:
    tg = to_grid.contiguous()
    fg = from_grid.contiguous()
    (kernel_values, kernel_frames, shape) = _kernel_layout(
        (grad_out, left, right), n_frames
    )
    kernel_grad_out, kernel_left, kernel_right = kernel_values
    gl = torch.empty_like(kernel_left, memory_format=torch.contiguous_format)
    gr = torch.empty_like(kernel_right, memory_format=torch.contiguous_format)
    try:
        _launch(
            _grid_pair_bwd_kernel,
            kernel_left,
            int(tg.shape[0]),
            kernel_frames,
            shape is not None,
            (
                kernel_grad_out,
                kernel_left,
                kernel_right,
                tg,
                fg,
                gl,
                gr,
                *_strides(kernel_grad_out, kernel_left, kernel_right, gl, gr),
            ),
            2,
        )
    except _NoViableConfig:
        gp, kernel_shape = _pack(kernel_grad_out, kernel_frames)
        lp, _ = _pack(kernel_left, kernel_frames)
        rp, _ = _pack(kernel_right, kernel_frames)
        gl, gr = _eager_packed("bwd", tg, fg, gp, lp, rp)
        gl = _unpack(gl, kernel_shape, kernel_frames)
        gr = _unpack(gr, kernel_shape, kernel_frames)
    return _restore_layout(gl, shape, n_frames), _restore_layout(gr, shape, n_frames)


_train_bwd_op = torch.library.triton_op(
    "sezm_triton::grid_pair_train_bwd",
    _train_bwd_impl,
    mutates_args=(),
)


@_train_bwd_op.register_fake
def _(grad_out, left, right, to_grid, from_grid, n_frames):
    del grad_out, to_grid, from_grid, n_frames
    return left.new_empty(left.shape), right.new_empty(right.shape)


def _train_bwd2_impl(
    h_gl: Tensor,
    h_gr: Tensor,
    grad_out: Tensor,
    left: Tensor,
    right: Tensor,
    to_grid: Tensor,
    from_grid: Tensor,
    n_frames: int,
) -> tuple[Tensor, Tensor, Tensor]:
    tg = to_grid.contiguous()
    fg = from_grid.contiguous()
    (kernel_values, kernel_frames, shape) = _kernel_layout(
        (h_gl, h_gr, grad_out, left, right), n_frames
    )
    kernel_h_gl, kernel_h_gr, kernel_grad_out, kernel_left, kernel_right = kernel_values
    ggo = torch.empty_like(kernel_grad_out, memory_format=torch.contiguous_format)
    g2l = torch.empty_like(kernel_left, memory_format=torch.contiguous_format)
    g2r = torch.empty_like(kernel_right, memory_format=torch.contiguous_format)
    kernel_args = (
        kernel_h_gl,
        kernel_h_gr,
        kernel_grad_out,
        kernel_left,
        kernel_right,
        tg,
        fg,
        ggo,
        g2l,
        g2r,
        *_strides(
            kernel_h_gl,
            kernel_h_gr,
            kernel_grad_out,
            kernel_left,
            kernel_right,
            ggo,
            g2l,
            g2r,
        ),
    )
    try:
        _launch(
            _grid_pair_bwd2_kernel,
            kernel_left,
            int(tg.shape[0]),
            kernel_frames,
            shape is not None,
            kernel_args,
            3,
        )
    except _NoViableConfig:
        hlp, kernel_shape = _pack(kernel_h_gl, kernel_frames)
        hrp, _ = _pack(kernel_h_gr, kernel_frames)
        gp, _ = _pack(kernel_grad_out, kernel_frames)
        lp, _ = _pack(kernel_left, kernel_frames)
        rp, _ = _pack(kernel_right, kernel_frames)
        ggo, g2l, g2r = _eager_packed("bwd2", tg, fg, hlp, hrp, gp, lp, rp)
        ggo = _unpack(ggo, kernel_shape, kernel_frames)
        g2l = _unpack(g2l, kernel_shape, kernel_frames)
        g2r = _unpack(g2r, kernel_shape, kernel_frames)
    return (
        _restore_layout(ggo, shape, n_frames),
        _restore_layout(g2l, shape, n_frames),
        _restore_layout(g2r, shape, n_frames),
    )


_train_bwd2_op = torch.library.triton_op(
    "sezm_triton::grid_pair_train_bwd2",
    _train_bwd2_impl,
    mutates_args=(),
)


@_train_bwd2_op.register_fake
def _(h_gl, h_gr, grad_out, left, right, to_grid, from_grid, n_frames):
    del h_gl, h_gr, grad_out, to_grid, from_grid, n_frames
    return (
        left.new_empty(left.shape),
        left.new_empty(left.shape),
        right.new_empty(right.shape),
    )


def _train_bwd_setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    grad_out, left, right, to_grid, from_grid, n_frames = inputs
    ctx.save_for_backward(grad_out, left, right, to_grid, from_grid)
    ctx.set_materialize_grads(False)
    ctx.n_frames = n_frames


def _train_bwd_backward(ctx: Any, h_gl: Tensor | None, h_gr: Tensor | None) -> tuple:
    """Second order of the pair product, force-loss regime.

    The backward is trilinear in ``(grad_out, left, right)``, so the three
    curvatures against the incoming gradient cotangents run as one kernel;
    the projectors are constants and carry none.
    """
    if h_gl is None and h_gr is None:
        return (None,) * 6
    grad_out, left, right, to_grid, from_grid = ctx.saved_tensors
    ggo, g2_left, g2_right = _train_bwd2_op(
        h_gl if h_gl is not None else torch.zeros_like(left),
        h_gr if h_gr is not None else torch.zeros_like(right),
        grad_out,
        left,
        right,
        to_grid,
        from_grid,
        int(ctx.n_frames),
    )
    return ggo, g2_left, g2_right, None, None, None


_train_bwd_op.register_autograd(
    _train_bwd_backward, setup_context=_train_bwd_setup_context
)


def _train_setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    left, right, to_grid, from_grid, n_frames = inputs
    ctx.save_for_backward(left, right, to_grid, from_grid)
    ctx.set_materialize_grads(False)
    ctx.n_frames = n_frames


def _train_backward(ctx: Any, grad_out: Tensor | None) -> tuple:
    left, right, to_grid, from_grid = ctx.saved_tensors
    if grad_out is None:
        return None, None, None, None, None
    g_left, g_right = _train_bwd_op(
        grad_out, left, right, to_grid, from_grid, int(ctx.n_frames)
    )
    return g_left, g_right, None, None, None


_train_op.register_autograd(_train_backward, setup_context=_train_setup_context)

# Under autocast the operands arrive in bfloat16 while the projector buffers
# stay float32, a mix the kernels must not consume half-and-half. Align every
# floating-point input to the autocast dtype exactly as the dense einsum
# composition does; the casts are recorded by autograd. Inert outside an
# autocast region.
_train_op.register_autocast("cuda", torch.bfloat16)
_train_bwd_op.register_autocast("cuda", torch.bfloat16)


def grid_pair_train(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    """
    Evaluate the pair product on frame-packed operands, differentiably.

    Parameters
    ----------
    left, right : torch.Tensor
        Coefficient operands with shape (N, D, F, K * C), where ``K`` is the
        frame count and the slot axis of the projectors runs (d, k).
    to_grid : torch.Tensor
        Coefficient-to-grid projector with shape (G, D * K).
    from_grid : torch.Tensor
        Grid-to-coefficient projector, transposed to shape (G, D * K).
    n_frames : int
        Frame count ``K`` packed along the trailing operand axis.

    Returns
    -------
    torch.Tensor
        Coefficient result with shape (N, D, F, K * C).
    """
    return _train_op(left, right, to_grid, from_grid, n_frames)
