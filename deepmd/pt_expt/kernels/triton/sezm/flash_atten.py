# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202, RUF005
"""Fused flash-attention edge->node aggregation for the SeZM/DPA4 SO(2) attention.

This module fuses the *entire* value-aggregation of the ``n_atten_head > 0``
branch of :class:`SO2Convolution` into a single destination-segmented Triton
kernel, for the shipped ``mmax == 1`` block-diagonal Wigner-D layout with
``atten_f_mix == atten_v_proj == atten_o_proj == False`` (the deployed DPA4
checkpoint).

Operation
---------
The eager attention aggregation is (per destination atom ``n``, degree row
``d`` of the packed ``(l, m)`` layout, hidden channel ``c = f * Cf + cf``)::

    out[n, d, c] = gate[n, f, h] * sum_{e: dst[e]=n}
                       alpha[e, f, h] * rescale[d] * RotBack_e(x_local)[d, c]

with ``h = cf // head_dim`` the attention head of channel ``c`` and

    RotBack_e(x_local)[d, c] = sum_m Dt[e, d, l^2+l+m] * x_local[e, f, (l,m), cf]

the block-diagonal ``local -> global`` rotation (block ``l`` of the transposed
Wigner-D). ``alpha`` is the destination-wise envelope-gated softmax weight (a
scalar per ``(edge, focus, head)``) and ``gate`` is the output-side head gate (a
scalar per ``(node, focus, head)``).

Because the softmax weight ``alpha`` and the output gate ``gate`` are cheap,
memory-negligible ``(E|N, F, H)`` scalars, only the *heavy* tensor work is fused
into the kernel: the block-diagonal ``rotate_back`` of the value, the per-edge
softmax weighting, the inverse-rotation rescale, and the destination-segmented
reduction. This kernel therefore computes the ungated aggregate

    pre_gate[n, d, c] = rescale[d] * sum_{e: dst[e]=n} alpha[e, f, h] * RotBack_e[d, c]

and the caller applies the node-level ``out = pre_gate * gate`` afterwards (its
backward is handled by autograd). This "two-step" split (a scalar segmented
softmax outside, the weighted rotate-back segment reduction fused inside) is
chosen over folding the online softmax into this kernel because the softmax
operates on scalar logits with no bandwidth cost, while the backward of the
fully-flashed variant would have to recompute or save the per-destination
softmax statistics -- for zero memory benefit. The chosen split keeps the heavy
kernel *linear* in ``(x_local, Dt, alpha)``, so the hand-written backward is
exact, saves no forward activation, and stays cheap.

The single fused forward removes the two largest transient edge tensors of the
eager path -- the rotate-back message ``x_message`` (E, D, C_wide) and the
``alpha``-weighted value ``weighted_value`` (E, D, C_wide) -- and the
``index_add`` round trip, which is the source of the end-to-end peak-memory
reduction.

Forward layout
--------------
One Triton program per destination node, reducing its edge segment through the
destination-sorted CSR view the step builds once and every segment consumer
shares (the traced edge list carries masked padding edges in arbitrary
destination order, so no sortedness invariant exists at this level): each
edge's block-diagonal ``rotate_back`` is assembled from the three retained
orders using per-degree register vectors (every reduced order is read exactly
once, no redundant gather), weighted by ``alpha``, and accumulated into a
``DIM``-row register tuple; the rescale is applied once per row at the final
store.  The contention-free CSR reduction is both faster than a per-edge
atomic scatter (which serializes on the colliding edges of each atom at
typical neighbor counts) and deterministic.

Backward layout
---------------
One Triton program per edge (no cross-edge accumulation, hence no atomics): it
reloads ``grad_pre_gate`` at the edge's destination, recomputes ``rotate_back``
from ``x_local`` and ``Dt``, and emits the exact per-edge gradients w.r.t.
``x_local`` (E, F, D_m, Cf), ``Dt`` (E, D, D; structural block-diagonal
non-zeros only, matching the shipped rotation kernels) and ``alpha`` (E, F, H).

Registration
------------
Forward and backward are functional ``torch.library.triton_op`` instances
(``mutates_args=()``) with registered fake kernels and an autograd formula whose
backward is itself a ``triton_op``. ``triton_op`` + ``wrap_triton`` (rather than
an opaque ``custom_op``) lets Inductor see through to the Triton kernels and bake
the cubins into the SeZM ``.pt2``, so the force graph
(``autograd.grad(energy, edge_vec)``) traces under ``make_fx`` and runs inside
the LAMMPS C++ runtime with no Python op registration. ``row_ptr`` / ``dst`` are
integer topology derived from the neighbor list (never the coordinates), so they
carry no gradient; ``rescale`` is a constant buffer and is likewise not
differentiated.
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

from .indexing import (
    build_m_major_index,
)
from .second_order import (
    accumulate,
)
from .tile_configs import (
    flash_bwd_block_config,
    flash_bwd_edge_config,
)

__all__ = [
    "FLASH_ATTEN_TRITON_AVAILABLE",
    "flash_atten_aggregate",
    "flash_atten_aggregate_reference",
]

try:
    import triton
    import triton.language as tl

    FLASH_ATTEN_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    FLASH_ATTEN_TRITON_AVAILABLE = False


# ======================================================================
# CSR row-pointer + per-row degree-map construction (integer, gradient-free)
# ======================================================================
# ======================================================================
# Eager reference / fallback implementation
# ======================================================================
def flash_atten_aggregate_reference(
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    dst: Tensor,
    n_nodes: int | torch.SymInt,
    lmax: int,
    n_head: int,
) -> Tensor:
    """Eager ground truth for :func:`flash_atten_aggregate` (block-diagonal).

    Parameters
    ----------
    x_local : Tensor
        Per-focus SO(2) features with shape ``(E, F, D_m, Cf)`` in the m-major
        ``mmax == 1`` reduced layout, ``D_m = 3 * lmax + 1``.
    wigner_dt : Tensor
        Transposed block-diagonal Wigner-D with shape ``(E, D, D)``,
        ``D = (lmax + 1) ** 2``.
    rescale : Tensor
        Inverse-rotation degree rescale aligned with the packed layout, ``(D,)``.
    alpha : Tensor
        Envelope-gated softmax weight with shape ``(E, F, H)``.
    dst : Tensor
        Destination node indices with shape ``(E,)``.
    n_nodes : int or torch.SymInt
        Number of destination nodes ``N``.
    lmax : int
        Maximum degree.
    n_head : int
        Number of attention heads ``H``.

    Returns
    -------
    Tensor
        Ungated aggregate ``pre_gate`` with shape ``(N, D, C_wide)``,
        ``C_wide = F * Cf``.
    """
    n_edge, n_focus, reduced_dim, focus_dim = x_local.shape
    dim = (int(lmax) + 1) ** 2
    c_wide = n_focus * focus_dim
    head_dim = focus_dim // int(n_head)
    coeff = build_m_major_index(int(lmax), 1, device=x_local.device)

    xl_std = x_local.transpose(1, 2).reshape(n_edge, reduced_dim, c_wide)
    dt_from_m = wigner_dt[:, :dim, :dim].index_select(2, coeff)  # (E, D, D_m)
    # Cast the constant fp64 ``rescale`` to the feature dtype so the reduction
    # stays in the caller's compute precision (no-op for fp64).
    resc = rescale.view(1, dim, 1).to(x_local.dtype)
    rb = torch.bmm(dt_from_m, xl_std) * resc  # (E, D, C_wide)
    # alpha (E, F, H) -> per-channel weight (E, C_wide) with c = f*Cf + h*head_dim + ch
    alpha_full = alpha.repeat_interleave(head_dim, dim=2).reshape(n_edge, c_wide)
    weighted = rb * alpha_full[:, None, :]
    out = x_local.new_zeros(n_nodes, dim, c_wide)
    out.index_add_(0, dst, weighted)
    return out


def _flash_atten_backward_reference(
    grad_pre_gate: Tensor,
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-form eager backward of :func:`flash_atten_aggregate_reference`.

    A closed form (not a nested ``autograd.grad``) is required because the
    backward operator carries no autograd formula and is dispatched under
    ``_AutoDispatchBelowAutograd`` when the SeZM ``.pt2`` force graph is replayed
    under :func:`torch.no_grad`, matching the discipline in ``radial_mix.py``.

    Returns ``(grad_x_local, grad_wigner, grad_alpha)``.
    """
    n_edge, n_focus, reduced_dim, focus_dim = x_local.shape
    dim = (int(lmax) + 1) ** 2
    c_wide = n_focus * focus_dim
    head_dim = focus_dim // int(n_head)
    coeff = build_m_major_index(int(lmax), 1, device=x_local.device)

    xl_std = x_local.transpose(1, 2).reshape(n_edge, reduced_dim, c_wide)
    dt_from_m = wigner_dt[:, :dim, :dim].index_select(2, coeff)  # (E, D, D_m)
    rb_pre = torch.bmm(dt_from_m, xl_std)  # (E, D, C_wide)
    # Cast ``rescale`` to the feature dtype (see the forward reference).
    resc = rescale.view(1, dim, 1).to(x_local.dtype)
    rb = rb_pre * resc
    alpha_full = alpha.repeat_interleave(head_dim, dim=2).reshape(n_edge, c_wide)

    grad_weighted = grad_pre_gate.index_select(0, dst)  # (E, D, C_wide)

    # grad w.r.t. alpha: sum over degree rows and head channels of grad*rb.
    grad_alpha_full = (grad_weighted * rb).sum(dim=1)  # (E, C_wide)
    grad_alpha = grad_alpha_full.reshape(n_edge, n_focus, int(n_head), head_dim).sum(
        dim=3
    )  # (E, F, H)

    # grad w.r.t. the rotate-back message, then split into x_local and Dt grads.
    grad_rb_pre = grad_weighted * alpha_full[:, None, :] * resc  # (E, D, C_wide)
    grad_xl_std = torch.bmm(dt_from_m.transpose(1, 2), grad_rb_pre)  # (E, D_m, C_wide)
    grad_x_local = grad_xl_std.reshape(
        n_edge, reduced_dim, n_focus, focus_dim
    ).transpose(1, 2)  # (E, F, D_m, Cf)

    grad_dt_from_m = torch.bmm(grad_rb_pre, xl_std.transpose(1, 2))  # (E, D, D_m)
    grad_block = wigner_dt.new_zeros(n_edge, dim, dim)
    grad_block.index_copy_(2, coeff, grad_dt_from_m)
    grad_wigner = torch.zeros_like(wigner_dt)
    grad_wigner[:, :dim, :dim] = grad_block
    return grad_x_local.contiguous(), grad_wigner, grad_alpha


# ======================================================================
# Triton kernels (mmax == 1; LMAX / layout are constexpr; channels vectorized)
# ======================================================================
if FLASH_ATTEN_TRITON_AVAILABLE:

    @triton.jit
    def _rotation_entry_offset(
        output_row,
        input_row,
        packed_index,
        row_stride,
        column_stride,
        PACKED: tl.constexpr,
    ):
        """Map a structural Wigner entry to dense or packed storage."""
        if PACKED:
            return packed_index * column_stride
        return output_row * row_stride + input_row * column_stride

    # The segmented forward carries a DIM-row register accumulator per
    # program, so low warp counts dominate; higher counts only pay off for
    # wide channel tiles.
    _FWD_CONFIGS = [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    _BWD_CONFIGS = [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ]

    @triton.autotune(configs=_FWD_CONFIGS, key=["C_wide", "PACKED"])
    @triton.jit
    def _flash_fwd_kernel(
        xl_ptr,
        dt_ptr,
        resc_ptr,
        w_ptr,
        order_ptr,
        row_ptr_ptr,
        out_ptr,
        n_node,
        C_wide,
        xl_se,
        xl_sf,
        xl_sr,
        xl_sc,
        dt_se,
        dt_sr,
        dt_sk,
        w_se,
        w_sf,
        w_sh,
        o_sn,
        o_sd,
        o_sc,
        LMAX: tl.constexpr,
        CF: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_C: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        """One program per node: indirect CSR segment reduction of the rotate-back.

        ``order`` lists edge ids sorted by destination and ``row_ptr`` holds
        the segment offsets, so program ``n`` reduces the edges
        ``order[row_ptr[n]..row_ptr[n+1]]`` -- an indirect, atomic-free and
        deterministic destination reduction that replaces the per-edge atomic
        scatter and accepts any edge order (the compiled SeZM graph keeps
        masked padding edges, so no sortedness invariant exists).  Per edge,
        each retained reduced order is read exactly once and neither the
        rotate-back message nor the weighted value is materialized to DRAM.
        Channels are the vectorized axis; ``c = f * Cf + cf`` decodes the
        per-focus ``x_local`` layout in place and the attention head is
        ``h = cf // head_dim``.  The ``DIM``-row accumulator lives in a
        loop-carried register tuple, so the rescale is applied once per row
        at the final store.
        """
        DIM: tl.constexpr = (LMAX + 1) * (LMAX + 1)

        node = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < C_wide
        beg = tl.load(row_ptr_ptr + node).to(tl.int64)
        end = tl.load(row_ptr_ptr + node + 1).to(tl.int64)

        # Channel decode c = f * Cf + cf, head h = cf // head_dim.  Masked
        # lanes clamp the focus index so pointer arithmetic stays in range.
        fv = tl.where(cmask, chan // CF, 0)
        cfv = chan % CF
        hv = cfv // HEAD_DIM
        xl_co = fv * xl_sf + cfv * xl_sc  # per-channel focus offset into x_local
        w_col = fv * w_sf + hv * w_sh  # per-channel (focus, head) offset into alpha

        acc = ()
        for _ in tl.static_range(DIM):
            acc = acc + (tl.zeros((BLOCK_C,), dtype=tl.float32),)

        for i in range(beg, end):
            edge = tl.load(order_ptr + i).to(tl.int64)
            wv = tl.load(w_ptr + edge * w_se + w_col, mask=cmask, other=0.0).to(
                tl.float32
            )
            new_acc = ()
            for l in tl.static_range(0, LMAX + 1):
                base = l * l
                r0 = base + l  # packed column of order m=0
                xl0 = tl.load(
                    xl_ptr + edge * xl_se + l * xl_sr + xl_co, mask=cmask, other=0.0
                ).to(tl.float32)
                if l >= 1:
                    xlm = tl.load(
                        xl_ptr + edge * xl_se + (LMAX + l) * xl_sr + xl_co,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    xlp = tl.load(
                        xl_ptr + edge * xl_se + (2 * LMAX + l) * xl_sr + xl_co,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                for j in tl.static_range(0, 2 * l + 1):
                    d = base + j  # full packed output row
                    offset0 = _rotation_entry_offset(
                        d, r0, base + j, dt_sr, dt_sk, PACKED
                    )
                    rb = tl.load(dt_ptr + edge * dt_se + offset0).to(tl.float32) * xl0
                    if l >= 1:
                        offset_m = _rotation_entry_offset(
                            d,
                            r0 - 1,
                            DIM + base - 1 + j,
                            dt_sr,
                            dt_sk,
                            PACKED,
                        )
                        offset_p = _rotation_entry_offset(
                            d,
                            r0 + 1,
                            2 * DIM + base - 2 + j,
                            dt_sr,
                            dt_sk,
                            PACKED,
                        )
                        rb += (
                            tl.load(dt_ptr + edge * dt_se + offset_m).to(tl.float32)
                            * xlm
                        )
                        rb += (
                            tl.load(dt_ptr + edge * dt_se + offset_p).to(tl.float32)
                            * xlp
                        )
                    # Loop-carried tuples require inline constexpr subscripts
                    # (the Triton frontend rejects composite index variables).
                    new_acc = new_acc + (acc[l * l + j] + rb * wv,)
            acc = new_acc

        for d in tl.static_range(DIM):
            resc = tl.load(resc_ptr + d).to(tl.float32)
            tl.store(
                out_ptr + node * o_sn + d * o_sd + chan * o_sc,
                acc[d] * resc,
                mask=cmask,
            )

    @triton.autotune(configs=_BWD_CONFIGS, key=["C_wide", "PACKED"])
    @triton.jit
    def _flash_bwd_kernel(
        gp_ptr,
        xl_ptr,
        dt_ptr,
        resc_ptr,
        w_ptr,
        dst_ptr,
        gxl_ptr,
        gdt_ptr,
        gw_ptr,
        n_edge,
        C_wide,
        gp_sn,
        gp_sd,
        gp_sc,
        xl_se,
        xl_sf,
        xl_sr,
        xl_sc,
        dt_se,
        dt_sr,
        dt_sk,
        w_se,
        w_sf,
        w_sh,
        gxl_se,
        gxl_sf,
        gxl_sr,
        gxl_sc,
        gdt_se,
        gdt_sr,
        gdt_sk,
        gw_se,
        gw_sf,
        gw_sh,
        LMAX: tl.constexpr,
        CF: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NFOCUS: tl.constexpr,
        NHEAD: tl.constexpr,
        BLOCK_C: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        """One program per edge: exact per-edge gradients of the fused forward.

        Reloads ``grad_pre_gate`` at the edge's destination, recomputes the
        block-diagonal ``rotate_back`` from ``x_local`` / ``Dt``, and stores
        ``grad_x_local``, ``grad_Dt`` (structural non-zeros) and ``grad_alpha``
        (reduced over each (focus, head) channel group). No cross-edge
        accumulation, hence no atomics.
        """
        DIM: tl.constexpr = (LMAX + 1) * (LMAX + 1)

        edge = tl.program_id(0).to(tl.int64)
        n = tl.load(dst_ptr + edge).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < C_wide
        fv = chan // CF
        cfv = chan % CF
        hv = cfv // HEAD_DIM
        xl_co = fv * xl_sf + cfv * xl_sc
        gxl_co = fv * gxl_sf + cfv * gxl_sc
        w_col = fv * w_sf + hv * w_sh
        grp = fv * NHEAD + hv  # (BLOCK_C,) flat (focus, head) group id

        wv = tl.load(w_ptr + edge * w_se + w_col, mask=cmask, other=0.0).to(tl.float32)
        gw_chan = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for l in tl.static_range(0, LMAX + 1):
            base = l * l
            r0 = base + l
            xl0 = tl.load(
                xl_ptr + edge * xl_se + l * xl_sr + xl_co, mask=cmask, other=0.0
            ).to(tl.float32)
            gxl0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            if l >= 1:
                xlm = tl.load(
                    xl_ptr + edge * xl_se + (LMAX + l) * xl_sr + xl_co,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                xlp = tl.load(
                    xl_ptr + edge * xl_se + (2 * LMAX + l) * xl_sr + xl_co,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                gxlm = tl.zeros((BLOCK_C,), dtype=tl.float32)
                gxlp = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for j in tl.static_range(0, 2 * l + 1):
                d = base + j
                offset0 = _rotation_entry_offset(d, r0, base + j, dt_sr, dt_sk, PACKED)
                grad_offset0 = _rotation_entry_offset(
                    d, r0, base + j, gdt_sr, gdt_sk, PACKED
                )
                resc = tl.load(resc_ptr + d).to(tl.float32)
                gpr = (
                    tl.load(
                        gp_ptr + n * gp_sn + d * gp_sd + chan * gp_sc,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    * resc
                )
                grad_rb = gpr * wv
                w0 = tl.load(dt_ptr + edge * dt_se + offset0).to(tl.float32)
                rb = w0 * xl0
                gxl0 += w0 * grad_rb
                tl.store(
                    gdt_ptr + edge * gdt_se + grad_offset0,
                    tl.sum(grad_rb * xl0).to(gdt_ptr.dtype.element_ty),
                )
                if l >= 1:
                    offset_m = _rotation_entry_offset(
                        d,
                        r0 - 1,
                        DIM + base - 1 + j,
                        dt_sr,
                        dt_sk,
                        PACKED,
                    )
                    offset_p = _rotation_entry_offset(
                        d,
                        r0 + 1,
                        2 * DIM + base - 2 + j,
                        dt_sr,
                        dt_sk,
                        PACKED,
                    )
                    grad_offset_m = _rotation_entry_offset(
                        d,
                        r0 - 1,
                        DIM + base - 1 + j,
                        gdt_sr,
                        gdt_sk,
                        PACKED,
                    )
                    grad_offset_p = _rotation_entry_offset(
                        d,
                        r0 + 1,
                        2 * DIM + base - 2 + j,
                        gdt_sr,
                        gdt_sk,
                        PACKED,
                    )
                    wm = tl.load(dt_ptr + edge * dt_se + offset_m).to(tl.float32)
                    wp = tl.load(dt_ptr + edge * dt_se + offset_p).to(tl.float32)
                    rb += wm * xlm + wp * xlp
                    gxlm += wm * grad_rb
                    gxlp += wp * grad_rb
                    tl.store(
                        gdt_ptr + edge * gdt_se + grad_offset_m,
                        tl.sum(grad_rb * xlm).to(gdt_ptr.dtype.element_ty),
                    )
                    tl.store(
                        gdt_ptr + edge * gdt_se + grad_offset_p,
                        tl.sum(grad_rb * xlp).to(gdt_ptr.dtype.element_ty),
                    )
                gw_chan += gpr * rb
            tl.store(
                gxl_ptr + edge * gxl_se + l * gxl_sr + gxl_co,
                gxl0.to(gxl_ptr.dtype.element_ty),
                mask=cmask,
            )
            if l >= 1:
                tl.store(
                    gxl_ptr + edge * gxl_se + (LMAX + l) * gxl_sr + gxl_co,
                    gxlm.to(gxl_ptr.dtype.element_ty),
                    mask=cmask,
                )
                tl.store(
                    gxl_ptr + edge * gxl_se + (2 * LMAX + l) * gxl_sr + gxl_co,
                    gxlp.to(gxl_ptr.dtype.element_ty),
                    mask=cmask,
                )

        for g in tl.static_range(0, NFOCUS * NHEAD):
            f = g // NHEAD
            h = g % NHEAD
            val = tl.sum(tl.where((grp == g) & cmask, gw_chan, 0.0))
            tl.store(
                gw_ptr + edge * gw_se + f * gw_sf + h * gw_sh,
                val.to(gw_ptr.dtype.element_ty),
            )

    @triton.jit
    def _flash_bwd_block_kernel(
        gp_ptr,  # (N, D, C) upstream gradient of the ungated aggregate
        xl_ptr,  # (E, F, D_m, Cf) local features
        dt_ptr,  # (E, D, D) transposed block-diagonal Wigner-D, contiguous
        resc_ptr,  # (D,) inverse-rotation rescale
        w_ptr,  # (E, F, H) attention weights, contiguous
        dst_ptr,  # (E,)
        gxl_ptr,  # (E, F, D_m, Cf) out
        gdt_ptr,  # (E, D, D) out (pre-zeroed, structural non-zeros written)
        gw_ptr,  # (E, F, H) out, contiguous
        n_edge,
        gp_sn,
        gp_sd,
        xl_se,
        xl_sf,
        xl_sr,
        xl_sc,
        gxl_se,
        gxl_sf,
        gxl_sr,
        gxl_sc,
        L: tl.constexpr,
        CF: tl.constexpr,
        CW: tl.constexpr,  # C_wide = F * Cf
        CP: tl.constexpr,  # next power of two >= CW (vector lane count)
        HEAD_DIM: tl.constexpr,
        NHEAD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Edge-block variant of the flash-attention backward.

        The per-edge kernel closes one cross-lane ``tl.sum`` per structural
        Wigner non-zero -- serialized warp shuffle-reduction chains that
        dominate its runtime on narrow hidden widths.  This variant processes
        ``BLOCK_E`` edges per program with the channel axis kept as the
        vector axis: every ``grad_Dt`` entry becomes one batched axis-1
        reduction of a ``(BLOCK_E, CP)`` tile, every ``grad_x_local`` term is
        a rank-1 vector FMA with the per-edge Wigner scalar broadcast over
        channels, and the per-edge scalars are loaded as coalesced
        ``(BLOCK_E,)`` vectors.  Channels are padded to the power-of-two lane
        count ``CP`` with masked lanes (no memory traffic, only register
        pressure, which the launch table absorbs with a smaller ``BLOCK_E``).

        The schedule wins only where the reduction overhead of the per-edge
        kernel dominates; :func:`~.tile_configs.flash_bwd_block_config` acts
        as the win list.
        """
        DIM: tl.constexpr = (L + 1) * (L + 1)
        NG: tl.constexpr = (CW // CF) * NHEAD  # flat (focus, head) group count
        PADDED: tl.constexpr = CP != CW

        pid = tl.program_id(0)
        offs_e = (pid * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
        e_mask = offs_e < n_edge
        eq = tl.where(e_mask, offs_e, 0)
        chan = tl.arange(0, CP)
        if PADDED:
            c_mask = chan < CW
            em = e_mask[:, None] & c_mask[None, :]
            # Masked lanes clamp their decode so pointer arithmetic stays valid.
            fv = tl.where(c_mask, chan // CF, 0)
            cfv = tl.where(c_mask, chan % CF, 0)
        else:
            em = e_mask[:, None]
            fv = chan // CF
            cfv = chan % CF
        hv = cfv // HEAD_DIM
        grp = fv * NHEAD + hv  # (CP,) flat (focus, head) group id

        dst = tl.load(dst_ptr + eq, mask=e_mask, other=0).to(tl.int64)
        # Attention weight broadcast to channels: w[e, f(c), h(c)].
        wv = tl.load(w_ptr + (eq * NG)[:, None] + grp[None, :], mask=em, other=0.0)

        xl_row = xl_ptr + (eq * xl_se)[:, None] + (fv * xl_sf + cfv * xl_sc)[None, :]
        gxl_row = (
            gxl_ptr + (eq * gxl_se)[:, None] + (fv * gxl_sf + cfv * gxl_sc)[None, :]
        )
        dt_base = dt_ptr + eq * DIM * DIM
        gdt_base = gdt_ptr + eq * DIM * DIM
        # The launcher passes a contiguous upstream gradient (channel stride 1).
        gp_row = gp_ptr + (dst * gp_sn)[:, None] + chan[None, :]

        gw_acc = tl.zeros((BLOCK_E, CP), dtype=tl.float32)

        for l in tl.static_range(0, L + 1):
            base = l * l
            r0 = base + l  # packed reduced column of order m = 0
            xl0 = tl.load(xl_row + l * xl_sr, mask=em, other=0.0)
            gxl0 = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            if l >= 1:
                xlm = tl.load(xl_row + (L + l) * xl_sr, mask=em, other=0.0)
                xlp = tl.load(xl_row + (2 * L + l) * xl_sr, mask=em, other=0.0)
                gxlm = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
                gxlp = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            for j in tl.static_range(0, 2 * l + 1):
                d = base + j
                resc = tl.load(resc_ptr + d)
                gpr = tl.load(gp_row + d * gp_sd, mask=em, other=0.0) * resc
                grad_rb = gpr * wv
                dt0 = tl.load(dt_base + d * DIM + r0, mask=e_mask, other=0.0)
                gxl0 += dt0[:, None] * grad_rb
                tl.store(
                    gdt_base + d * DIM + r0,
                    tl.sum(grad_rb * xl0, axis=1),
                    mask=e_mask,
                )
                rb = dt0[:, None] * xl0
                if l >= 1:
                    dtm = tl.load(dt_base + d * DIM + (r0 - 1), mask=e_mask, other=0.0)
                    dtp = tl.load(dt_base + d * DIM + (r0 + 1), mask=e_mask, other=0.0)
                    gxlm += dtm[:, None] * grad_rb
                    gxlp += dtp[:, None] * grad_rb
                    tl.store(
                        gdt_base + d * DIM + (r0 - 1),
                        tl.sum(grad_rb * xlm, axis=1),
                        mask=e_mask,
                    )
                    tl.store(
                        gdt_base + d * DIM + (r0 + 1),
                        tl.sum(grad_rb * xlp, axis=1),
                        mask=e_mask,
                    )
                    rb += dtm[:, None] * xlm + dtp[:, None] * xlp
                gw_acc += gpr * rb
            tl.store(gxl_row + l * gxl_sr, gxl0, mask=em)
            if l >= 1:
                tl.store(gxl_row + (L + l) * gxl_sr, gxlm, mask=em)
                tl.store(gxl_row + (2 * L + l) * gxl_sr, gxlp, mask=em)

        # grad_alpha: reduce gw_acc over each (focus, head) channel group.
        for g in tl.static_range(NG):
            val = tl.sum(tl.where((grp == g)[None, :] & em, gw_acc, 0.0), axis=1)
            tl.store(gw_ptr + eq * NG + g, val, mask=e_mask)

    @triton.autotune(configs=_FWD_CONFIGS, key=["C_wide", "PACKED"])
    @triton.jit
    def _flash_2nd_gather_kernel(
        xl_ptr,
        hxl_ptr,
        dt_ptr,
        hdt_ptr,
        resc_ptr,
        w_ptr,
        hw_ptr,
        order_ptr,
        row_ptr_ptr,
        out_ptr,
        n_node,
        C_wide,
        xl_se,
        xl_sf,
        xl_sr,
        xl_sc,
        hxl_se,
        hxl_sf,
        hxl_sr,
        hxl_sc,
        dt_se,
        dt_sr,
        dt_sk,
        hdt_se,
        hdt_sr,
        hdt_sk,
        w_se,
        w_sf,
        w_sh,
        hw_se,
        hw_sf,
        hw_sh,
        o_sn,
        o_sd,
        o_sc,
        LMAX: tl.constexpr,
        CF: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_C: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        """Output-cotangent term of the aggregation's second order, one pass.

        The aggregation is trilinear in ``(x_local, Dt, alpha)``, so the
        cotangent of its upstream gradient is the sum of the forward evaluated
        at each incoming cotangent in turn:

            d_gout = resc * sum_e [ Dt (alpha h_x + h_alpha x) + h_Dt (alpha x) ]

        One CSR pass over the destination segment replaces the three forward
        re-entries of the substitution form; the structure mirrors the
        forward kernel with the rotate-back operand widened to the mixed
        combinations above.
        """
        DIM: tl.constexpr = (LMAX + 1) * (LMAX + 1)

        node = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < C_wide
        beg = tl.load(row_ptr_ptr + node).to(tl.int64)
        end = tl.load(row_ptr_ptr + node + 1).to(tl.int64)

        fv = tl.where(cmask, chan // CF, 0)
        cfv = chan % CF
        hv = cfv // HEAD_DIM
        xl_co = fv * xl_sf + cfv * xl_sc
        hxl_co = fv * hxl_sf + cfv * hxl_sc
        w_col = fv * w_sf + hv * w_sh
        hw_col = fv * hw_sf + hv * hw_sh

        acc = ()
        for _ in tl.static_range(DIM):
            acc = acc + (tl.zeros((BLOCK_C,), dtype=tl.float32),)

        for i in range(beg, end):
            edge = tl.load(order_ptr + i).to(tl.int64)
            wv = tl.load(w_ptr + edge * w_se + w_col, mask=cmask, other=0.0).to(
                tl.float32
            )
            hwv = tl.load(hw_ptr + edge * hw_se + hw_col, mask=cmask, other=0.0).to(
                tl.float32
            )
            new_acc = ()
            for l in tl.static_range(0, LMAX + 1):
                base = l * l
                r0 = base + l
                xl0 = tl.load(
                    xl_ptr + edge * xl_se + l * xl_sr + xl_co, mask=cmask, other=0.0
                ).to(tl.float32)
                hx0 = tl.load(
                    hxl_ptr + edge * hxl_se + l * hxl_sr + hxl_co,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                # Mixed rotate-back operands: v = alpha h_x + h_alpha x pairs
                # with Dt, and u = alpha x pairs with h_Dt.
                v0 = wv * hx0 + hwv * xl0
                u0 = wv * xl0
                if l >= 1:
                    xlm = tl.load(
                        xl_ptr + edge * xl_se + (LMAX + l) * xl_sr + xl_co,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    xlp = tl.load(
                        xl_ptr + edge * xl_se + (2 * LMAX + l) * xl_sr + xl_co,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    hxm = tl.load(
                        hxl_ptr + edge * hxl_se + (LMAX + l) * hxl_sr + hxl_co,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    hxp = tl.load(
                        hxl_ptr + edge * hxl_se + (2 * LMAX + l) * hxl_sr + hxl_co,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    vm = wv * hxm + hwv * xlm
                    vp = wv * hxp + hwv * xlp
                    um = wv * xlm
                    up = wv * xlp
                for j in tl.static_range(0, 2 * l + 1):
                    d = base + j
                    offset0 = _rotation_entry_offset(
                        d, r0, base + j, dt_sr, dt_sk, PACKED
                    )
                    h_offset0 = _rotation_entry_offset(
                        d, r0, base + j, hdt_sr, hdt_sk, PACKED
                    )
                    dt0 = tl.load(dt_ptr + edge * dt_se + offset0).to(tl.float32)
                    hdt0 = tl.load(hdt_ptr + edge * hdt_se + h_offset0).to(tl.float32)
                    rb = dt0 * v0 + hdt0 * u0
                    if l >= 1:
                        offset_m = _rotation_entry_offset(
                            d,
                            r0 - 1,
                            DIM + base - 1 + j,
                            dt_sr,
                            dt_sk,
                            PACKED,
                        )
                        offset_p = _rotation_entry_offset(
                            d,
                            r0 + 1,
                            2 * DIM + base - 2 + j,
                            dt_sr,
                            dt_sk,
                            PACKED,
                        )
                        h_offset_m = _rotation_entry_offset(
                            d,
                            r0 - 1,
                            DIM + base - 1 + j,
                            hdt_sr,
                            hdt_sk,
                            PACKED,
                        )
                        h_offset_p = _rotation_entry_offset(
                            d,
                            r0 + 1,
                            2 * DIM + base - 2 + j,
                            hdt_sr,
                            hdt_sk,
                            PACKED,
                        )
                        dtm = tl.load(dt_ptr + edge * dt_se + offset_m).to(tl.float32)
                        dtp = tl.load(dt_ptr + edge * dt_se + offset_p).to(tl.float32)
                        hdtm = tl.load(hdt_ptr + edge * hdt_se + h_offset_m).to(
                            tl.float32
                        )
                        hdtp = tl.load(hdt_ptr + edge * hdt_se + h_offset_p).to(
                            tl.float32
                        )
                        rb += dtm * vm + dtp * vp + hdtm * um + hdtp * up
                    new_acc = new_acc + (acc[l * l + j] + rb,)
            acc = new_acc

        for d in tl.static_range(DIM):
            resc = tl.load(resc_ptr + d).to(tl.float32)
            tl.store(
                out_ptr + node * o_sn + d * o_sd + chan * o_sc,
                acc[d] * resc,
                mask=cmask,
            )

    @triton.autotune(configs=_BWD_CONFIGS, key=["C_wide", "PACKED"])
    @triton.jit
    def _flash_2nd_edge_kernel(
        gp_ptr,
        xl_ptr,
        hxl_ptr,
        dt_ptr,
        hdt_ptr,
        resc_ptr,
        w_ptr,
        hw_ptr,
        dst_ptr,
        dxl_ptr,
        ddt_ptr,
        dw_ptr,
        n_edge,
        C_wide,
        gp_sn,
        gp_sd,
        gp_sc,
        xl_se,
        xl_sf,
        xl_sr,
        xl_sc,
        hxl_se,
        hxl_sf,
        hxl_sr,
        hxl_sc,
        dt_se,
        dt_sr,
        dt_sk,
        hdt_se,
        hdt_sr,
        hdt_sk,
        w_se,
        w_sf,
        w_sh,
        hw_se,
        hw_sf,
        hw_sh,
        dxl_se,
        dxl_sf,
        dxl_sr,
        dxl_sc,
        ddt_se,
        ddt_sr,
        ddt_sk,
        dw_se,
        dw_sf,
        dw_sh,
        LMAX: tl.constexpr,
        CF: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NFOCUS: tl.constexpr,
        NHEAD: tl.constexpr,
        BLOCK_C: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        """Edge-side terms of the aggregation's second order, one pass.

        Differentiating the first-order backward at the incoming cotangents
        ``(h_x, h_Dt, h_alpha)`` gives, per edge with ``gpr`` the rescaled
        upstream gradient at the destination,

            d_x[k]    = sum_d gpr * ( Dt[d,k] h_alpha + h_Dt[d,k] alpha )
            d_Dt[d,k] = sum_c gpr * ( h_alpha x[k] + alpha h_x[k] )
            d_alpha   = sum_d sum_c gpr * ( (Dt h_x)[d] + (h_Dt x)[d] )

        One pass over the structural non-zeros replaces the three backward
        re-entries of the substitution form; the structure mirrors the
        per-edge backward kernel with every operand paired against its
        cotangent.
        """
        DIM: tl.constexpr = (LMAX + 1) * (LMAX + 1)

        edge = tl.program_id(0).to(tl.int64)
        n = tl.load(dst_ptr + edge).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < C_wide
        fv = chan // CF
        cfv = chan % CF
        hv = cfv // HEAD_DIM
        xl_co = fv * xl_sf + cfv * xl_sc
        hxl_co = fv * hxl_sf + cfv * hxl_sc
        dxl_co = fv * dxl_sf + cfv * dxl_sc
        w_col = fv * w_sf + hv * w_sh
        hw_col = fv * hw_sf + hv * hw_sh
        grp = fv * NHEAD + hv

        wv = tl.load(w_ptr + edge * w_se + w_col, mask=cmask, other=0.0).to(tl.float32)
        hwv = tl.load(hw_ptr + edge * hw_se + hw_col, mask=cmask, other=0.0).to(
            tl.float32
        )
        dw_chan = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for l in tl.static_range(0, LMAX + 1):
            base = l * l
            r0 = base + l
            xl0 = tl.load(
                xl_ptr + edge * xl_se + l * xl_sr + xl_co, mask=cmask, other=0.0
            ).to(tl.float32)
            hx0 = tl.load(
                hxl_ptr + edge * hxl_se + l * hxl_sr + hxl_co, mask=cmask, other=0.0
            ).to(tl.float32)
            dxl0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            if l >= 1:
                xlm = tl.load(
                    xl_ptr + edge * xl_se + (LMAX + l) * xl_sr + xl_co,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                xlp = tl.load(
                    xl_ptr + edge * xl_se + (2 * LMAX + l) * xl_sr + xl_co,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                hxm = tl.load(
                    hxl_ptr + edge * hxl_se + (LMAX + l) * hxl_sr + hxl_co,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                hxp = tl.load(
                    hxl_ptr + edge * hxl_se + (2 * LMAX + l) * hxl_sr + hxl_co,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                dxlm = tl.zeros((BLOCK_C,), dtype=tl.float32)
                dxlp = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for j in tl.static_range(0, 2 * l + 1):
                d = base + j
                offset0 = _rotation_entry_offset(d, r0, base + j, dt_sr, dt_sk, PACKED)
                h_offset0 = _rotation_entry_offset(
                    d, r0, base + j, hdt_sr, hdt_sk, PACKED
                )
                d_offset0 = _rotation_entry_offset(
                    d, r0, base + j, ddt_sr, ddt_sk, PACKED
                )
                resc = tl.load(resc_ptr + d).to(tl.float32)
                gpr = (
                    tl.load(
                        gp_ptr + n * gp_sn + d * gp_sd + chan * gp_sc,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    * resc
                )
                grad_ha = gpr * hwv  # pairs with Dt for d_x
                grad_a = gpr * wv  # pairs with h_Dt for d_x
                dt0 = tl.load(dt_ptr + edge * dt_se + offset0).to(tl.float32)
                hdt0 = tl.load(hdt_ptr + edge * hdt_se + h_offset0).to(tl.float32)
                dxl0 += dt0 * grad_ha + hdt0 * grad_a
                tl.store(
                    ddt_ptr + edge * ddt_se + d_offset0,
                    tl.sum(gpr * (hwv * xl0 + wv * hx0)).to(ddt_ptr.dtype.element_ty),
                )
                rb_mix = dt0 * hx0 + hdt0 * xl0
                if l >= 1:
                    offset_m = _rotation_entry_offset(
                        d,
                        r0 - 1,
                        DIM + base - 1 + j,
                        dt_sr,
                        dt_sk,
                        PACKED,
                    )
                    offset_p = _rotation_entry_offset(
                        d,
                        r0 + 1,
                        2 * DIM + base - 2 + j,
                        dt_sr,
                        dt_sk,
                        PACKED,
                    )
                    h_offset_m = _rotation_entry_offset(
                        d,
                        r0 - 1,
                        DIM + base - 1 + j,
                        hdt_sr,
                        hdt_sk,
                        PACKED,
                    )
                    h_offset_p = _rotation_entry_offset(
                        d,
                        r0 + 1,
                        2 * DIM + base - 2 + j,
                        hdt_sr,
                        hdt_sk,
                        PACKED,
                    )
                    d_offset_m = _rotation_entry_offset(
                        d,
                        r0 - 1,
                        DIM + base - 1 + j,
                        ddt_sr,
                        ddt_sk,
                        PACKED,
                    )
                    d_offset_p = _rotation_entry_offset(
                        d,
                        r0 + 1,
                        2 * DIM + base - 2 + j,
                        ddt_sr,
                        ddt_sk,
                        PACKED,
                    )
                    dtm = tl.load(dt_ptr + edge * dt_se + offset_m).to(tl.float32)
                    dtp = tl.load(dt_ptr + edge * dt_se + offset_p).to(tl.float32)
                    hdtm = tl.load(hdt_ptr + edge * hdt_se + h_offset_m).to(tl.float32)
                    hdtp = tl.load(hdt_ptr + edge * hdt_se + h_offset_p).to(tl.float32)
                    dxlm += dtm * grad_ha + hdtm * grad_a
                    dxlp += dtp * grad_ha + hdtp * grad_a
                    tl.store(
                        ddt_ptr + edge * ddt_se + d_offset_m,
                        tl.sum(gpr * (hwv * xlm + wv * hxm)).to(
                            ddt_ptr.dtype.element_ty
                        ),
                    )
                    tl.store(
                        ddt_ptr + edge * ddt_se + d_offset_p,
                        tl.sum(gpr * (hwv * xlp + wv * hxp)).to(
                            ddt_ptr.dtype.element_ty
                        ),
                    )
                    rb_mix += dtm * hxm + dtp * hxp + hdtm * xlm + hdtp * xlp
                dw_chan += gpr * rb_mix
            tl.store(
                dxl_ptr + edge * dxl_se + l * dxl_sr + dxl_co,
                dxl0.to(dxl_ptr.dtype.element_ty),
                mask=cmask,
            )
            if l >= 1:
                tl.store(
                    dxl_ptr + edge * dxl_se + (LMAX + l) * dxl_sr + dxl_co,
                    dxlm.to(dxl_ptr.dtype.element_ty),
                    mask=cmask,
                )
                tl.store(
                    dxl_ptr + edge * dxl_se + (2 * LMAX + l) * dxl_sr + dxl_co,
                    dxlp.to(dxl_ptr.dtype.element_ty),
                    mask=cmask,
                )

        for g in tl.static_range(0, NFOCUS * NHEAD):
            f = g // NHEAD
            h = g % NHEAD
            val = tl.sum(tl.where((grp == g) & cmask, dw_chan, 0.0))
            tl.store(
                dw_ptr + edge * dw_se + f * dw_sf + h * dw_sh,
                val.to(dw_ptr.dtype.element_ty),
            )


# ======================================================================
# Tile helper + zero-edge guard
# ======================================================================
def _tile_channels(channels: int) -> int:
    """Smallest power-of-two channel tile of at least 16 covering ``channels``."""
    tile = 16
    while tile < int(channels):
        tile *= 2
    return tile


def _has_no_edges(n_edge) -> bool:
    """Return true only for a concrete zero-edge call (SymInt-safe guard)."""
    return type(n_edge) is int and n_edge == 0


def _rotation_strides(rotation: Tensor) -> tuple[bool, int, int, int]:
    """Return the storage mode and strides for dense or packed rotations."""
    if rotation.dim() == 2:
        return True, rotation.stride(0), 0, rotation.stride(1)
    if rotation.dim() == 3:
        return False, rotation.stride(0), rotation.stride(1), rotation.stride(2)
    raise ValueError(
        "rotation must have shape (E, D, D) or (E, 3 * D - 2), "
        f"got rank {rotation.dim()}"
    )


# ======================================================================
# Triton launch wrappers
# ======================================================================
def _launch_forward(
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    n_nodes: int | torch.SymInt,
    lmax: int,
    n_head: int,
) -> Tensor:
    n_edge, n_focus, _reduced_dim, focus_dim = x_local.shape
    dim = (int(lmax) + 1) ** 2
    c_wide = n_focus * focus_dim
    packed, dt_se, dt_sr, dt_sk = _rotation_strides(wigner_dt)
    # The segment reduction accumulates in float32 registers regardless of
    # the input precision and writes each output row exactly once.
    out = torch.empty(n_nodes, dim, c_wide, dtype=torch.float32, device=x_local.device)
    if _has_no_edges(n_edge):
        return out.zero_().to(x_local.dtype)
    wrap_triton(_flash_fwd_kernel)[(n_nodes,)](
        x_local,
        wigner_dt,
        rescale,
        alpha,
        order,
        row_ptr,
        out,
        n_nodes,
        c_wide,
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        x_local.stride(3),
        dt_se,
        dt_sr,
        dt_sk,
        alpha.stride(0),
        alpha.stride(1),
        alpha.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LMAX=int(lmax),
        CF=focus_dim,
        HEAD_DIM=focus_dim // int(n_head),
        BLOCK_C=_tile_channels(c_wide),
        PACKED=packed,
    )
    return out.to(x_local.dtype)


def _launch_backward(
    grad_pre_gate: Tensor,
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> tuple[Tensor, Tensor, Tensor]:
    n_edge, n_focus, _reduced_dim, focus_dim = x_local.shape
    c_wide = n_focus * focus_dim
    grad_x_local = torch.empty_like(x_local)
    grad_wigner = torch.zeros_like(wigner_dt, memory_format=torch.contiguous_format)
    grad_alpha = torch.empty_like(alpha)
    packed, dt_se, dt_sr, dt_sk = _rotation_strides(wigner_dt)
    _, gdt_se, gdt_sr, gdt_sk = _rotation_strides(grad_wigner)
    if _has_no_edges(n_edge):
        return grad_x_local, grad_wigner, grad_alpha
    # The edge-block schedule engages on swept-and-winning (C_wide, lmax)
    # keys; every other shape keeps the per-edge kernel.  The branch resolves
    # at trace time, so exactly one kernel reaches the compiled graph.
    block_cfg = None if packed else flash_bwd_block_config(int(c_wide), int(lmax))
    if block_cfg is not None:
        block_e, warps, stages = block_cfg
        wrap_triton(_flash_bwd_block_kernel)[(triton.cdiv(n_edge, block_e),)](
            grad_pre_gate,
            x_local,
            wigner_dt.contiguous(),
            rescale,
            alpha,
            dst,
            grad_x_local,
            grad_wigner,
            grad_alpha,
            n_edge,
            grad_pre_gate.stride(0),
            grad_pre_gate.stride(1),
            x_local.stride(0),
            x_local.stride(1),
            x_local.stride(2),
            x_local.stride(3),
            grad_x_local.stride(0),
            grad_x_local.stride(1),
            grad_x_local.stride(2),
            grad_x_local.stride(3),
            L=int(lmax),
            CF=focus_dim,
            CW=c_wide,
            CP=triton.next_power_of_2(c_wide),
            HEAD_DIM=focus_dim // int(n_head),
            NHEAD=int(n_head),
            BLOCK_E=block_e,
            num_warps=warps,
            num_stages=stages,
        )
        return grad_x_local, grad_wigner, grad_alpha
    edge_cfg = flash_bwd_edge_config(int(c_wide), int(lmax))
    edge_kernel = _flash_bwd_kernel if edge_cfg is None else _flash_bwd_kernel.fn
    edge_launch = wrap_triton(edge_kernel)[(n_edge,)]
    edge_kwargs = (
        {}
        if edge_cfg is None
        else {
            "num_warps": edge_cfg[0],
            "num_stages": edge_cfg[1],
        }
    )
    edge_launch(
        grad_pre_gate,
        x_local,
        wigner_dt,
        rescale,
        alpha,
        dst,
        grad_x_local,
        grad_wigner,
        grad_alpha,
        n_edge,
        c_wide,
        grad_pre_gate.stride(0),
        grad_pre_gate.stride(1),
        grad_pre_gate.stride(2),
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        x_local.stride(3),
        dt_se,
        dt_sr,
        dt_sk,
        alpha.stride(0),
        alpha.stride(1),
        alpha.stride(2),
        grad_x_local.stride(0),
        grad_x_local.stride(1),
        grad_x_local.stride(2),
        grad_x_local.stride(3),
        gdt_se,
        gdt_sr,
        gdt_sk,
        grad_alpha.stride(0),
        grad_alpha.stride(1),
        grad_alpha.stride(2),
        LMAX=int(lmax),
        CF=focus_dim,
        HEAD_DIM=focus_dim // int(n_head),
        NFOCUS=n_focus,
        NHEAD=int(n_head),
        BLOCK_C=_tile_channels(c_wide),
        PACKED=packed,
        **edge_kwargs,
    )
    return grad_x_local, grad_wigner, grad_alpha


# ======================================================================
# Dispatch helpers (triton on CUDA float, eager otherwise)
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        FLASH_ATTEN_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _forward_impl(
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> Tensor:
    if not _use_triton(x_local):
        return flash_atten_aggregate_reference(
            x_local,
            wigner_dt,
            rescale,
            alpha,
            dst,
            row_ptr.shape[0] - 1,
            int(lmax),
            int(n_head),
        )
    # ``x_local`` is passed with its native (possibly transposed) strides -- the
    # kernel addresses it through the stride arguments, and preserving the layout
    # keeps the backward's ``grad_x_local`` stride-compatible with the stock
    # ``rotate_back_block_so2`` path so the downstream SO(2) backward reshapes
    # stay viewable under symbolic (make_fx / AOT) restride. ``N`` is taken from
    # ``row_ptr.shape`` (a SymInt) so the ``natoms`` axis is never specialized.
    return _launch_forward(
        x_local,
        wigner_dt,
        rescale.contiguous(),
        alpha.contiguous(),
        order.contiguous(),
        row_ptr.contiguous(),
        row_ptr.shape[0] - 1,
        int(lmax),
        int(n_head),
    )


def _backward_impl(
    grad_pre_gate: Tensor,
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> tuple[Tensor, Tensor, Tensor]:
    # The per-edge backward addresses destinations through ``dst`` alone and does
    # not read the CSR view. ``order`` / ``row_ptr`` are carried so that the
    # operator's own autograd formula, which re-enters the segmented forward for
    # the second-order term, can reach them: ``setup_context`` only sees the
    # arguments of the operator it belongs to.
    if not _use_triton(x_local):
        return _flash_atten_backward_reference(
            grad_pre_gate,
            x_local,
            wigner_dt,
            rescale,
            alpha,
            dst,
            int(lmax),
            int(n_head),
        )
    # Keep ``x_local``'s native strides so ``grad_x_local = empty_like(x_local)``
    # matches the stock ``rotate_back_block_so2`` backward layout (see the
    # forward note); only ``grad_pre_gate`` is made contiguous for coalesced
    # gather reads.
    return _launch_backward(
        grad_pre_gate.contiguous(),
        x_local,
        wigner_dt,
        rescale.contiguous(),
        alpha.contiguous(),
        dst.contiguous(),
        int(lmax),
        int(n_head),
    )


def _second_order_gather_impl(
    x_local: Tensor,
    h_x: Tensor,
    wigner_dt: Tensor,
    h_wigner: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    h_alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> Tensor:
    """Output-cotangent term of the second order in one CSR pass."""
    if not _use_triton(x_local):
        n_node = row_ptr.shape[0] - 1
        return (
            flash_atten_aggregate_reference(
                h_x, wigner_dt, rescale, alpha, dst, n_node, int(lmax), int(n_head)
            )
            + flash_atten_aggregate_reference(
                x_local, h_wigner, rescale, alpha, dst, n_node, int(lmax), int(n_head)
            )
            + flash_atten_aggregate_reference(
                x_local,
                wigner_dt,
                rescale,
                h_alpha,
                dst,
                n_node,
                int(lmax),
                int(n_head),
            )
        )
    n_node = row_ptr.shape[0] - 1
    n_focus, focus_dim = x_local.shape[1], x_local.shape[3]
    c_wide = n_focus * focus_dim
    dim = (int(lmax) + 1) ** 2
    packed, dt_se, dt_sr, dt_sk = _rotation_strides(wigner_dt)
    h_packed, hdt_se, hdt_sr, hdt_sk = _rotation_strides(h_wigner)
    if h_packed != packed:
        raise ValueError("rotation and its cotangent must use the same storage layout")
    out = x_local.new_empty(n_node, dim, c_wide)
    if _has_no_edges(x_local.shape[0]):
        return out.zero_()
    alpha = alpha.contiguous()
    h_alpha = h_alpha.contiguous()
    rescale = rescale.contiguous()
    wrap_triton(_flash_2nd_gather_kernel)[(n_node,)](
        x_local,
        h_x,
        wigner_dt,
        h_wigner,
        rescale,
        alpha,
        h_alpha,
        order.contiguous(),
        row_ptr.contiguous(),
        out,
        n_node,
        c_wide,
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        x_local.stride(3),
        h_x.stride(0),
        h_x.stride(1),
        h_x.stride(2),
        h_x.stride(3),
        dt_se,
        dt_sr,
        dt_sk,
        hdt_se,
        hdt_sr,
        hdt_sk,
        alpha.stride(0),
        alpha.stride(1),
        alpha.stride(2),
        h_alpha.stride(0),
        h_alpha.stride(1),
        h_alpha.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LMAX=int(lmax),
        CF=focus_dim,
        HEAD_DIM=focus_dim // int(n_head),
        BLOCK_C=_tile_channels(c_wide),
        PACKED=packed,
    )
    return out


def _second_order_edge_impl(
    grad_pre_gate: Tensor,
    x_local: Tensor,
    h_x: Tensor,
    wigner_dt: Tensor,
    h_wigner: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    h_alpha: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Edge-side terms of the second order in one pass over the non-zeros."""
    if not _use_triton(x_local):
        gx_w, _, ga_w = _flash_atten_backward_reference(
            grad_pre_gate,
            x_local,
            h_wigner,
            rescale,
            alpha,
            dst,
            int(lmax),
            int(n_head),
        )
        gx_a, gdt_a, _ = _flash_atten_backward_reference(
            grad_pre_gate,
            x_local,
            wigner_dt,
            rescale,
            h_alpha,
            dst,
            int(lmax),
            int(n_head),
        )
        _, gdt_x, ga_x = _flash_atten_backward_reference(
            grad_pre_gate, h_x, wigner_dt, rescale, alpha, dst, int(lmax), int(n_head)
        )
        return gx_w + gx_a, gdt_a + gdt_x, ga_w + ga_x
    n_edge = x_local.shape[0]
    n_focus, focus_dim = x_local.shape[1], x_local.shape[3]
    c_wide = n_focus * focus_dim
    d_x = torch.empty_like(x_local)
    # The kernel writes only the structural non-zeros (three columns per
    # degree block), so the Wigner gradient must start from zeros.
    d_dt = torch.zeros_like(wigner_dt, memory_format=torch.contiguous_format)
    d_alpha = torch.empty_like(alpha)
    packed, dt_se, dt_sr, dt_sk = _rotation_strides(wigner_dt)
    h_packed, hdt_se, hdt_sr, hdt_sk = _rotation_strides(h_wigner)
    d_packed, ddt_se, ddt_sr, ddt_sk = _rotation_strides(d_dt)
    if h_packed != packed or d_packed != packed:
        raise ValueError("rotation tensors must use the same storage layout")
    if _has_no_edges(n_edge):
        return d_x, d_dt, d_alpha
    grad_pre_gate = grad_pre_gate.contiguous()
    alpha = alpha.contiguous()
    h_alpha = h_alpha.contiguous()
    rescale = rescale.contiguous()
    wrap_triton(_flash_2nd_edge_kernel)[(n_edge,)](
        grad_pre_gate,
        x_local,
        h_x,
        wigner_dt,
        h_wigner,
        rescale,
        alpha,
        h_alpha,
        dst.contiguous(),
        d_x,
        d_dt,
        d_alpha,
        n_edge,
        c_wide,
        grad_pre_gate.stride(0),
        grad_pre_gate.stride(1),
        grad_pre_gate.stride(2),
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        x_local.stride(3),
        h_x.stride(0),
        h_x.stride(1),
        h_x.stride(2),
        h_x.stride(3),
        dt_se,
        dt_sr,
        dt_sk,
        hdt_se,
        hdt_sr,
        hdt_sk,
        alpha.stride(0),
        alpha.stride(1),
        alpha.stride(2),
        h_alpha.stride(0),
        h_alpha.stride(1),
        h_alpha.stride(2),
        d_x.stride(0),
        d_x.stride(1),
        d_x.stride(2),
        d_x.stride(3),
        ddt_se,
        ddt_sr,
        ddt_sk,
        d_alpha.stride(0),
        d_alpha.stride(1),
        d_alpha.stride(2),
        LMAX=int(lmax),
        CF=focus_dim,
        HEAD_DIM=focus_dim // int(n_head),
        NFOCUS=n_focus,
        NHEAD=int(n_head),
        BLOCK_C=_tile_channels(c_wide),
        PACKED=packed,
    )
    return d_x, d_dt, d_alpha


# ======================================================================
# Functional triton_op + fake + autograd registration
# ======================================================================
_flash_op = torch.library.triton_op(
    "sezm_triton::flash_atten_aggregate", mutates_args=()
)(_forward_impl)

_flash_bwd_op = torch.library.triton_op(
    "sezm_triton::flash_atten_aggregate_bwd", mutates_args=()
)(_backward_impl)

_flash_2nd_gather_op = torch.library.triton_op(
    "sezm_triton::flash_atten_aggregate_2nd_gather", mutates_args=()
)(_second_order_gather_impl)

_flash_2nd_edge_op = torch.library.triton_op(
    "sezm_triton::flash_atten_aggregate_2nd_edge", mutates_args=()
)(_second_order_edge_impl)


@_flash_2nd_gather_op.register_fake
def _(
    x_local,
    h_x,
    wigner_dt,
    h_wigner,
    rescale,
    alpha,
    h_alpha,
    order,
    row_ptr,
    dst,
    lmax,
    n_head,
):
    n_focus, focus_dim = x_local.shape[1], x_local.shape[3]
    dim = (int(lmax) + 1) ** 2
    return x_local.new_empty(row_ptr.shape[0] - 1, dim, n_focus * focus_dim)


@_flash_2nd_edge_op.register_fake
def _(
    grad_pre_gate,
    x_local,
    h_x,
    wigner_dt,
    h_wigner,
    rescale,
    alpha,
    h_alpha,
    dst,
    lmax,
    n_head,
):
    return (
        torch.empty_like(x_local),
        torch.empty_like(wigner_dt),
        torch.empty_like(alpha),
    )


@_flash_op.register_fake
def _(x_local, wigner_dt, rescale, alpha, order, row_ptr, dst, lmax, n_head):
    n_focus = x_local.shape[1]
    focus_dim = x_local.shape[3]
    dim = (int(lmax) + 1) ** 2
    # ``N`` is derived from ``row_ptr`` (not an int arg) so the dynamic ``natoms``
    # axis survives ``torch.export`` without specialization.
    return x_local.new_empty(row_ptr.shape[0] - 1, dim, n_focus * focus_dim)


@_flash_bwd_op.register_fake
def _(
    grad_pre_gate, x_local, wigner_dt, rescale, alpha, order, row_ptr, dst, lmax, n_head
):
    return (
        torch.empty_like(x_local),
        torch.empty_like(wigner_dt),
        torch.empty_like(alpha),
    )


def _setup_context(ctx, inputs, output):
    x_local, wigner_dt, rescale, alpha, order, row_ptr, dst, lmax, n_head = inputs
    ctx.save_for_backward(x_local, wigner_dt, rescale, alpha, order, row_ptr, dst)
    ctx.lmax = lmax
    ctx.n_head = n_head


def _backward(ctx, grad_out):
    x_local, wigner_dt, rescale, alpha, order, row_ptr, dst = ctx.saved_tensors
    grad_x_local, grad_wigner, grad_alpha = _flash_bwd_op(
        grad_out.contiguous(),
        x_local,
        wigner_dt,
        rescale,
        alpha,
        order,
        row_ptr,
        dst,
        ctx.lmax,
        ctx.n_head,
    )
    # inputs: x_local, wigner_dt, rescale, alpha, order, row_ptr, dst, lmax,
    # n_head. rescale is a constant buffer; order/row_ptr/dst are integer
    # topology.
    return grad_x_local, grad_wigner, None, grad_alpha, None, None, None, None, None


def _bwd_setup_context(ctx, inputs, output):
    (
        grad_pre_gate,
        x_local,
        wigner_dt,
        rescale,
        alpha,
        order,
        row_ptr,
        dst,
        lmax,
        n_head,
    ) = inputs
    ctx.save_for_backward(
        grad_pre_gate, x_local, wigner_dt, rescale, alpha, order, row_ptr, dst
    )
    ctx.lmax = lmax
    ctx.n_head = n_head


def _bwd_backward(ctx, grad_grad_x_local, grad_grad_wigner, grad_grad_alpha):
    """Second order of the aggregation, trilinear in ``(x_local, Dt, alpha)``.

    Substituting all three cotangents in one backward call would create cross
    terms, because each component of the backward still depends on two of the
    three operands. Each call therefore substitutes exactly one cotangent and
    contributes the two components that actually see it.
    """
    grad_out, x_local, wigner, rescale, alpha, order, row_ptr, dst = ctx.saved_tensors
    lmax, n_head = ctx.lmax, ctx.n_head
    h_x, h_wigner, h_alpha = grad_grad_x_local, grad_grad_wigner, grad_grad_alpha
    if h_x is None and h_wigner is None and h_alpha is None:
        return (None,) * 10

    if h_x is not None and h_wigner is not None and h_alpha is not None:
        # The force-loss trace materializes every cotangent, so this is the
        # production branch: one gather pass and one edge pass replace the
        # three forward and three backward re-entries of the substitution
        # form below.
        grad_grad_out = None
        if ctx.needs_input_grad[0]:
            grad_grad_out = _flash_2nd_gather_op(
                x_local,
                h_x,
                wigner,
                h_wigner,
                rescale,
                alpha,
                h_alpha,
                order,
                row_ptr,
                dst,
                lmax,
                n_head,
            )
        grad_x, grad_wigner, grad_alpha = _flash_2nd_edge_op(
            grad_out,
            x_local,
            h_x,
            wigner,
            h_wigner,
            rescale,
            alpha,
            h_alpha,
            dst,
            lmax,
            n_head,
        )
        return (
            grad_grad_out,
            grad_x,
            grad_wigner,
            None,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
        )

    def forward(x_arg: Tensor, w_arg: Tensor, a_arg: Tensor) -> Tensor:
        return _flash_op(
            x_arg, w_arg, rescale, a_arg, order, row_ptr, dst, lmax, n_head
        )

    def backward(x_arg: Tensor, w_arg: Tensor, a_arg: Tensor) -> tuple[Tensor, ...]:
        return _flash_bwd_op(
            grad_out, x_arg, w_arg, rescale, a_arg, order, row_ptr, dst, lmax, n_head
        )

    grad_grad_out: Tensor | None = None
    grad_x: Tensor | None = None
    grad_wigner: Tensor | None = None
    grad_alpha: Tensor | None = None
    # Each substitution costs one forward for the output-cotangent term; a graph
    # that does not propagate past this point skips all of them.
    needs_grad_out = ctx.needs_input_grad[0]

    if h_x is not None:
        if needs_grad_out:
            grad_grad_out = forward(h_x, wigner, alpha)
        _, term_wigner, term_alpha = backward(h_x, wigner, alpha)
        grad_wigner = accumulate(grad_wigner, term_wigner)
        grad_alpha = accumulate(grad_alpha, term_alpha)
    if h_wigner is not None:
        if needs_grad_out:
            grad_grad_out = accumulate(grad_grad_out, forward(x_local, h_wigner, alpha))
        term_x, _, term_alpha = backward(x_local, h_wigner, alpha)
        grad_x = accumulate(grad_x, term_x)
        grad_alpha = accumulate(grad_alpha, term_alpha)
    if h_alpha is not None:
        if needs_grad_out:
            grad_grad_out = accumulate(grad_grad_out, forward(x_local, wigner, h_alpha))
        term_x, term_wigner, _ = backward(x_local, wigner, h_alpha)
        grad_x = accumulate(grad_x, term_x)
        grad_wigner = accumulate(grad_wigner, term_wigner)

    # inputs: grad_pre_gate, x_local, wigner_dt, rescale, alpha, order, row_ptr,
    # dst, lmax, n_head.
    return (
        grad_grad_out,
        grad_x,
        grad_wigner,
        None,
        grad_alpha,
        None,
        None,
        None,
        None,
        None,
    )


_flash_op.register_autograd(_backward, setup_context=_setup_context)
_flash_bwd_op.register_autograd(_bwd_backward, setup_context=_bwd_setup_context)

# Under AMP the local features arrive in bfloat16 while the Wigner-D and the
# rescale buffers are still float32; the autocast rule aligns them the way the
# built-in matmuls do and is inert outside an autocast region.
_flash_op.register_autocast("cuda", torch.bfloat16)


# ======================================================================
# Public API
# ======================================================================
def flash_atten_aggregate(
    x_local: Tensor,
    wigner_dt: Tensor,
    rescale: Tensor,
    alpha: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
    lmax: int,
    n_head: int,
) -> Tensor:
    """Fused block-diagonal rotate-back + envelope-softmax weighting + edge scatter.

    Computes the ungated attention aggregate

        ``pre_gate[n, d, c] = rescale[d] *
            sum_{e: dst[e]=n} alpha[e, f, h] * RotBack_e(x_local)[d, c]``

    for the ``mmax == 1`` block-diagonal layout, equivalent to the eager
    ``rotate_back -> rescale -> value-reshape -> alpha-weight -> index_add`` chain
    of :class:`SO2Convolution` (the caller applies the node-level output gate
    ``out = pre_gate * gate`` afterwards).

    Parameters
    ----------
    x_local : Tensor
        Per-focus SO(2) features with shape ``(E, F, D_m, Cf)``.
    wigner_dt : Tensor
        Transposed block-diagonal Wigner-D with shape ``(E, D, D)``.
    rescale : Tensor
        Inverse-rotation degree rescale with shape ``(D,)``.
    alpha : Tensor
        Envelope-gated softmax weight with shape ``(E, F, H)``.
    order : Tensor
        Destination-sorted edge permutation with shape ``(E,)``, the segment
        order of the forward reduction. The active descriptor backend builds
        it once with ``cached_edge_csr`` and every segment consumer shares it.
    row_ptr : Tensor
        Row offsets with shape ``(N + 1,)`` matching ``order``; its length also
        carries the (SymInt) node count ``N`` for the output allocation and the
        fake kernel, so the ``natoms`` axis is never specialized.
    dst : Tensor
        Destination node indices with shape ``(E,)`` (the backward gather
        index).
    lmax : int
        Maximum degree.
    n_head : int
        Number of attention heads ``H``.

    Returns
    -------
    Tensor
        Ungated aggregate with shape ``(N, D, C_wide)``, ``C_wide = F * Cf``.
    """
    return _flash_op(
        x_local,
        wigner_dt,
        rescale,
        alpha,
        order,
        row_ptr,
        dst,
        int(lmax),
        int(n_head),
    )
