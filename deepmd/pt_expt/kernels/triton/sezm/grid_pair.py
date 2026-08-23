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
    def _grid_pair_fwd_kernel(
        left_ptr,
        right_ptr,
        tg_ptr,
        fg_ptr,
        out_ptr,
        n_pair,
        n_grid,
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
        base = pair * P_DIM * C_ALL

        p_hi = tl.arange(0, P_HI)
        hi_mask = p_hi < P_DIM
        off_hi = base + p_hi[:, None] * C_ALL + c_idx[None, :]
        m_hi = hi_mask[:, None] & c_mask[None, :]
        lv_hi = tl.load(left_ptr + off_hi, mask=m_hi, other=0.0)
        rv_hi = tl.load(right_ptr + off_hi, mask=m_hi, other=0.0)
        acc_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        if P_LO > 0:
            p_lo = P_HI + tl.arange(0, P_LO)
            lo_mask = p_lo < P_DIM
            off_lo = base + p_lo[:, None] * C_ALL + c_idx[None, :]
            m_lo = lo_mask[:, None] & c_mask[None, :]
            lv_lo = tl.load(left_ptr + off_lo, mask=m_lo, other=0.0)
            rv_lo = tl.load(right_ptr + off_lo, mask=m_lo, other=0.0)
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

        tl.store(out_ptr + off_hi, acc_hi.to(out_ptr.dtype.element_ty), mask=m_hi)
        if P_LO > 0:
            tl.store(out_ptr + off_lo, acc_lo.to(out_ptr.dtype.element_ty), mask=m_lo)

    @triton.jit
    def _grid_pair_bwd_kernel(
        go_ptr,
        left_ptr,
        right_ptr,
        tg_ptr,
        fg_ptr,
        gl_ptr,
        gr_ptr,
        n_pair,
        n_grid,
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
        base = pair * P_DIM * C_ALL

        p_hi = tl.arange(0, P_HI)
        hi_mask = p_hi < P_DIM
        off_hi = base + p_hi[:, None] * C_ALL + c_idx[None, :]
        m_hi = hi_mask[:, None] & c_mask[None, :]
        lv_hi = tl.load(left_ptr + off_hi, mask=m_hi, other=0.0)
        rv_hi = tl.load(right_ptr + off_hi, mask=m_hi, other=0.0)
        go_hi = tl.load(go_ptr + off_hi, mask=m_hi, other=0.0)
        gl_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        gr_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        if P_LO > 0:
            p_lo = P_HI + tl.arange(0, P_LO)
            lo_mask = p_lo < P_DIM
            off_lo = base + p_lo[:, None] * C_ALL + c_idx[None, :]
            m_lo = lo_mask[:, None] & c_mask[None, :]
            lv_lo = tl.load(left_ptr + off_lo, mask=m_lo, other=0.0)
            rv_lo = tl.load(right_ptr + off_lo, mask=m_lo, other=0.0)
            go_lo = tl.load(go_ptr + off_lo, mask=m_lo, other=0.0)
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

        tl.store(gl_ptr + off_hi, gl_hi.to(gl_ptr.dtype.element_ty), mask=m_hi)
        tl.store(gr_ptr + off_hi, gr_hi.to(gr_ptr.dtype.element_ty), mask=m_hi)
        if P_LO > 0:
            tl.store(gl_ptr + off_lo, gl_lo.to(gl_ptr.dtype.element_ty), mask=m_lo)
            tl.store(gr_ptr + off_lo, gr_lo.to(gr_ptr.dtype.element_ty), mask=m_lo)

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
        n_pair,
        n_grid,
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
        base = pair * P_DIM * C_ALL

        p_hi = tl.arange(0, P_HI)
        hi_mask = p_hi < P_DIM
        off_hi = base + p_hi[:, None] * C_ALL + c_idx[None, :]
        m_hi = hi_mask[:, None] & c_mask[None, :]
        lv_hi = tl.load(left_ptr + off_hi, mask=m_hi, other=0.0)
        rv_hi = tl.load(right_ptr + off_hi, mask=m_hi, other=0.0)
        go_hi = tl.load(go_ptr + off_hi, mask=m_hi, other=0.0)
        hl_hi = tl.load(hgl_ptr + off_hi, mask=m_hi, other=0.0)
        hr_hi = tl.load(hgr_ptr + off_hi, mask=m_hi, other=0.0)
        ao_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        al_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        ar_hi = tl.zeros((P_HI, C_BLK), dtype=tl.float32)
        if P_LO > 0:
            p_lo = P_HI + tl.arange(0, P_LO)
            lo_mask = p_lo < P_DIM
            off_lo = base + p_lo[:, None] * C_ALL + c_idx[None, :]
            m_lo = lo_mask[:, None] & c_mask[None, :]
            lv_lo = tl.load(left_ptr + off_lo, mask=m_lo, other=0.0)
            rv_lo = tl.load(right_ptr + off_lo, mask=m_lo, other=0.0)
            go_lo = tl.load(go_ptr + off_lo, mask=m_lo, other=0.0)
            hl_lo = tl.load(hgl_ptr + off_lo, mask=m_lo, other=0.0)
            hr_lo = tl.load(hgr_ptr + off_lo, mask=m_lo, other=0.0)
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

        tl.store(ggo_ptr + off_hi, ao_hi.to(ggo_ptr.dtype.element_ty), mask=m_hi)
        tl.store(g2l_ptr + off_hi, al_hi.to(g2l_ptr.dtype.element_ty), mask=m_hi)
        tl.store(g2r_ptr + off_hi, ar_hi.to(g2r_ptr.dtype.element_ty), mask=m_hi)
        if P_LO > 0:
            tl.store(ggo_ptr + off_lo, ao_lo.to(ggo_ptr.dtype.element_ty), mask=m_lo)
            tl.store(g2l_ptr + off_lo, al_lo.to(g2l_ptr.dtype.element_ty), mask=m_lo)
            tl.store(g2r_ptr + off_lo, ar_lo.to(g2r_ptr.dtype.element_ty), mask=m_lo)


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


_LAUNCH_CACHE: dict[tuple, tuple[int, int, int, int]] = {}


def _launch(kernel, packed: Tensor, n_grid: int, args: tuple, n_acc: int) -> None:
    """Launch with the largest tile the register and shared budgets admit.

    The channel block is capped so the ``n_acc`` fp32 accumulator tiles
    ``(P_PAD, C_BLK)`` stay register-resident (a spilled accumulator is
    read-modify-written through local memory once per grid block, which
    dominated the three-accumulator second-order kernel before the cap).
    The exact shared footprint additionally depends on Triton's internal
    staging (dot operand buffers, transpose scratch), so candidates are
    tried from the most to the least aggressive and the first that compiles
    is cached per ``(kernel, slots, channels, dtype)``.
    """
    n_pair, p_dim, c_per = packed.shape
    # The slot axis is covered by the largest power of two below the count
    # plus an optional low segment for the remainder, so 147 (degree six)
    # pads to 128 + 32 and 75 (degree four) to 64 + 16 instead of the next
    # power of two. The high segment keeps the tensor-core minimum of 16.
    p_hi = max(16, 1 << (p_dim.bit_length() - 1))
    p_lo = _next_pow2(p_dim - p_hi) if p_dim > p_hi else 0
    p_eff = p_hi + p_lo
    c_top = min(64, _next_pow2(c_per), max(16, _next_pow2(4096 // (n_acc * p_eff))))
    key = (kernel.fn.__name__, p_dim, c_per, packed.dtype)
    candidates = [
        (c_blk, block_g, stages)
        for c_blk in (c_top, 32, 16)
        if c_blk <= c_top
        for block_g, stages in ((64, 2), (32, 2), (32, 1), (16, 1))
    ]
    if key in _LAUNCH_CACHE:
        candidates = [_LAUNCH_CACHE[key]]
    for c_blk, block_g, stages in candidates:
        grid = (n_pair, (c_per + c_blk - 1) // c_blk)
        try:
            kernel[grid](
                *args,
                n_pair,
                n_grid=n_grid,
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
    lp, shape = _pack(left, n_frames)
    rp, _ = _pack(right, n_frames)
    tg = to_grid.contiguous()
    fg = from_grid.contiguous()
    out = torch.empty_like(lp)
    try:
        _launch(_grid_pair_fwd_kernel, lp, int(tg.shape[0]), (lp, rp, tg, fg, out), 1)
    except _NoViableConfig:
        (out,) = _eager_packed("fwd", tg, fg, lp, rp)
    return _unpack(out, shape, n_frames)


_train_op = torch.library.custom_op(
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
    gp, shape = _pack(grad_out, n_frames)
    lp, _ = _pack(left, n_frames)
    rp, _ = _pack(right, n_frames)
    tg = to_grid.contiguous()
    fg = from_grid.contiguous()
    gl = torch.empty_like(lp)
    gr = torch.empty_like(rp)
    try:
        _launch(
            _grid_pair_bwd_kernel,
            lp,
            int(tg.shape[0]),
            (gp, lp, rp, tg, fg, gl, gr),
            2,
        )
    except _NoViableConfig:
        gl, gr = _eager_packed("bwd", tg, fg, gp, lp, rp)
    return _unpack(gl, shape, n_frames), _unpack(gr, shape, n_frames)


_train_bwd_op = torch.library.custom_op(
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
    hlp, shape = _pack(h_gl, n_frames)
    hrp, _ = _pack(h_gr, n_frames)
    gp, _ = _pack(grad_out, n_frames)
    lp, _ = _pack(left, n_frames)
    rp, _ = _pack(right, n_frames)
    tg = to_grid.contiguous()
    fg = from_grid.contiguous()
    ggo = torch.empty_like(lp)
    g2l = torch.empty_like(lp)
    g2r = torch.empty_like(rp)
    try:
        _launch(
            _grid_pair_bwd2_kernel,
            lp,
            int(tg.shape[0]),
            (hlp, hrp, gp, lp, rp, tg, fg, ggo, g2l, g2r),
            3,
        )
    except _NoViableConfig:
        ggo, g2l, g2r = _eager_packed("bwd2", tg, fg, hlp, hrp, gp, lp, rp)
    return (
        _unpack(ggo, shape, n_frames),
        _unpack(g2l, shape, n_frames),
        _unpack(g2r, shape, n_frames),
    )


_train_bwd2_op = torch.library.custom_op(
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
