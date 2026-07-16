# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Fused graph-native environment convolution for the DPA1 (``se_atten``)
descriptor, in either concat or strip tebd-input mode.

The graph lower represents the neighbor list as a flat edge stream of ``E``
edges (``edge_index = [src, dst]``, ``edge_vec = r_src - r_dst``) rather than a
padded ``(node, nnei)`` grid, and replaces the neighbor-axis reduction with a
``segment_sum`` over edge centers (``dst``). For the attention-free
(``attn_layer == 0``) path the per-edge descriptor tail is

    ``h2  = act(z2) * idt + resnet(h1)``       (last embedding layer + resnet)
    ``gg  = h2 * (1 + tt[idx] * sw)``           (strip type-pair gate; concat: gg = h2)
    ``gg  = gg * edge_mask``                    (drop padding edges)
    ``gr[dst, k, c] += rr[k] * gg[c]``          (outer product + segment_sum)

where ``act`` is the last layer's activation (``tanh`` or ``silu``), ``z2`` the
last embedding pre-activation, ``h1`` the penultimate activation feeding the
resnet, ``idt`` the per-channel timestep, and ``rr`` the ``(s, s*x/r, s*y/r,
s*z/r)`` per-edge environment matrix. The two tebd-input modes differ only in
the gate (``gated``): concat carries the type feature through the embedding
input (no gate); strip factorizes it into the type-pair table ``tt`` gathered
per edge by ``idx`` and scaled by the per-edge switch ``sw``, matching the dense
``se_conv`` gate. The ``1 / nnei`` normalization of ``gr`` is applied by the
descriptor after this operator, so the operator returns the unnormalized moment.

Layout and strategy
-------------------
- **Node-parallel forward via a CSR segment reduction.** The operator builds a
  destination-sorted topology internally (``argsort`` on ``dst`` plus
  ``searchsorted`` for the per-node segment offsets -- integer, gradient-free),
  so the forward is one program per node: it streams the node's edges (a
  contiguous segment of the sorted ``order``) in ``BLOCK_E``-edge blocks, forms
  ``gg`` in registers, accumulates the four moment rows and writes ``gr`` (shape
  ``(N, 4, ng)``) exactly once. This is contention-free -- unlike a per-edge
  ``atomic_add`` scatter, whose ~nnei-way ``dst`` collisions serialize at
  production neighbor densities -- so it matches the dense ``se_conv`` register
  reduction while streaming only real edges (no ``sel`` padding). ``edge_mask``
  must gate ``gg``: ``edge_env_mat`` normalizes by the per-type mean, so padding
  edges carry a *nonzero* ``rr`` and would otherwise contaminate the reduction.
- **Gather backward, edge-parallel.** The backward reads the upstream moment
  gradient once per edge from ``grad_gr[dst]`` (a gather) and writes ``dz2`` /
  ``dh1`` / ``drr`` per edge, so it needs neither the sorted topology nor
  atomics.
- **Arbitrary channel widths via padding.** ``tl.arange`` requires a
  power-of-two bound; the channel axis is carried at ``next_pow2(ng)`` and the
  surplus lanes are masked on every load and store, and the ``BLOCK_E`` segment
  tail is masked by ``pos < end``.

The two embedding GEMMs stay on cuBLAS (fp32 has no ``tl.dot`` tensor-core path)
and the ``ng x ng x 4`` Gram contraction that forms the final descriptor is
likewise left on cuBLAS; only the memory-bound tail is fused.

The operator is a ``triton_op`` and composes under ``make_fx`` / ``torch.export``
-- the graph lower is the pt_expt ``.pt2`` export target, whose forward traces
``forward_common_lower_graph`` over the edge schema.
"""

from __future__ import (
    annotations,
)

import logging

import torch
import torch.nn.functional as F
from torch import (
    Tensor,
)
from torch.library import (
    triton_op,
    wrap_triton,
)

from deepmd.kernels.autotune import (
    register_autotuner,
)
from deepmd.kernels.triton.dpa1.activation import (
    TRITON_AVAILABLE,
)
from deepmd.kernels.triton.dpa1.tile_configs import (
    has_edge_config,
    register_edge_config,
    resolve_edge_config,
)
from deepmd.kernels.utils import (
    triton_infer_level,
)

log = logging.getLogger(__name__)

__all__ = [
    "concat_gate_placeholders",
    "edge_conv",
]


# ======================================================================
# Eager reference / fallback implementation
# ======================================================================
def _edge_conv_reference(
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    dst: Tensor,
    edge_mask: Tensor,
    n_node: int,
    resnet_mult: int,
    act: int,
    gated: int,
) -> Tensor:
    """Eager ground truth for the graph environment convolution.

    ``gated`` selects the tebd-input mode: ``1`` (strip) applies the type-pair
    gate ``gg = h2 * (1 + tt[idx] * sw)``; ``0`` (concat) skips it (the type
    feature enters upstream through the embedding input), so ``tt`` / ``idx`` /
    ``sw`` are ignored. Returns the unnormalized moment ``gr`` (n_node, 4, ng).
    """
    activated = torch.tanh(z2) if act == 0 else F.silu(z2)
    h2 = activated * idt
    if resnet_mult == 2:
        h2 = h2 + torch.cat([h1, h1], dim=-1)
    elif resnet_mult == 1:
        h2 = h2 + h1
    if gated:
        gg_t = tt.index_select(0, idx.reshape(-1))  # (E, ng)
        h2 = h2 * (1.0 + gg_t * sw[:, None])
    gg = h2 * edge_mask.to(z2.dtype)[:, None]  # (E, ng)
    # outer product (E, 4, ng): rr[:, k] * gg[:, c]
    outer = rr[:, :, None] * gg[:, None, :]
    gr = torch.zeros((n_node, 4, z2.shape[-1]), dtype=z2.dtype, device=z2.device)
    gr.index_add_(0, dst, outer)
    return gr


def _edge_conv_reference_backward(
    grad_gr: Tensor,
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    dst: Tensor,
    edge_mask: Tensor,
    n_node: int,
    resnet_mult: int,
    act: int,
    gated: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Closed-form gradient of :func:`_edge_conv_reference` w.r.t. the
    coordinate-bearing inputs ``(z2, h1, sw, rr)``.

    A closed form (rather than a nested ``torch.autograd.grad``) is used so the
    fallback composes under ``make_fx`` / ``torch.export``; the expressions
    mirror the Triton backward kernel. ``gated == 0`` (concat) drops the type
    gate, so ``fac == 1`` and ``grad_sw == 0``.
    """
    if act == 0:
        a = torch.tanh(z2)
        act_grad = 1.0 - a * a
    else:
        s = torch.sigmoid(z2)
        a = z2 * s
        act_grad = s * (1.0 + z2 * (1.0 - s))
    h2 = a * idt
    if resnet_mult == 2:
        h2 = h2 + torch.cat([h1, h1], dim=-1)
    elif resnet_mult == 1:
        h2 = h2 + h1
    m = edge_mask.to(z2.dtype)[:, None]
    if gated:
        gg_t = tt.index_select(0, idx.reshape(-1))  # (E, ng)
        fac = 1.0 + gg_t * sw[:, None]  # d gg / d h2
    else:
        fac = torch.ones_like(h2)
    gg = h2 * fac * m
    # gather the upstream moment gradient at each edge's center: (E, 4, ng)
    g_e = grad_gr.index_select(0, dst)
    # d gg = sum_k rr[k] * grad_gr[dst, k]; grad_rr[k] = sum_c gg[c] * grad_gr[dst, k]
    dgg = (g_e * rr[:, :, None]).sum(dim=1)  # (E, ng)
    grad_rr = (g_e * gg[:, None, :]).sum(dim=-1)  # (E, 4)
    grad_h2 = dgg * fac * m
    grad_z2 = grad_h2 * idt * act_grad
    # fac = 1 + gg_t * sw, so d gg / d sw = h2 * gg_t (zero without the gate).
    if gated:
        grad_sw = (dgg * h2 * gg_t).sum(dim=-1) * m[:, 0]
    else:
        grad_sw = torch.zeros_like(sw)
    if resnet_mult == 2:
        h1d = h1.shape[-1]
        grad_h1 = grad_h2[:, :h1d] + grad_h2[:, h1d : 2 * h1d]
    elif resnet_mult == 1:
        grad_h1 = grad_h2
    else:
        grad_h1 = torch.zeros_like(h1)
    return grad_z2, grad_h1, grad_sw, grad_rr


# ======================================================================
# Triton kernels
# ======================================================================
if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    from deepmd.kernels.triton.dpa1.activation import (
        activation,
        activation_grad,
    )

    @triton.jit
    def _edge_conv_fwd_kernel(
        z2_ptr,  # (E, NG) last embedding pre-activation
        h1_ptr,  # (E, H1) penultimate activation (resnet source)
        idt_ptr,  # (NG,) last-layer timestep
        tt_ptr,  # (P, NG) type-pair embedding table (strip gate)
        idx_ptr,  # (E,) per-edge type-pair row index (strip gate)
        sw_ptr,  # (E,) per-edge smooth cutoff (strip gate)
        rr_ptr,  # (E, 4) per-edge environment matrix
        mask_ptr,  # (E,) valid-edge mask (float 1/0)
        order_ptr,  # (E,) edge permutation grouping edges by center (dst-sorted)
        segptr_ptr,  # (N + 1,) CSR offsets of each node's segment in ``order``
        gr_ptr,  # (N, 4, NG) accumulated moments
        NG: tl.constexpr,
        NGP: tl.constexpr,
        H1: tl.constexpr,
        RESNET_MULT: tl.constexpr,
        ACT: tl.constexpr,
        GATED: tl.constexpr,
        BE: tl.constexpr,
    ):
        """Reduce one node's four moment rows over its CSR edge segment.

        One program per node: it streams the node's edges -- contiguous in the
        dst-sorted ``order`` between ``segptr[node]`` and ``segptr[node + 1]`` --
        in ``BE``-edge blocks, forms ``gg`` in registers and accumulates the four
        moment rows, writing ``gr[node]`` exactly once. This is the
        contention-free analogue of an atomic scatter (whose ~nnei-way ``dst``
        collisions serialize): the reduction is register-local, like the dense
        ``se_conv``, but streams only real edges (no ``sel`` padding). Edge
        indices from ``order`` are int64 (``e * NG`` overflows int32 on large
        systems); the channel mask ``rc < NG`` guards the padded width and the
        segment mask ``pos < end`` guards the ``BE``-block tail.

        ``GATED`` selects the tebd-input mode: ``1`` (strip) applies the
        type-pair gate ``gg = h2 * (1 + tt[idx] * sw)`` (the type feature
        gathered inline through ``idx``); ``0`` (concat) skips it (``gg = h2``),
        since concat feeds the type feature through the embedding input.
        """
        node = tl.program_id(0)
        rc = tl.arange(0, NGP)
        cm = rc < NG
        idt = tl.load(idt_ptr + rc, mask=cm, other=0.0)
        beg = tl.load(segptr_ptr + node)
        end = tl.load(segptr_ptr + node + 1)
        acc_ty = z2_ptr.dtype.element_ty
        acc0 = tl.zeros((NGP,), dtype=acc_ty)
        acc1 = tl.zeros((NGP,), dtype=acc_ty)
        acc2 = tl.zeros((NGP,), dtype=acc_ty)
        acc3 = tl.zeros((NGP,), dtype=acc_ty)
        for i0 in range(beg, end, BE):
            pos = i0 + tl.arange(0, BE)
            seg = pos < end
            e = tl.load(order_ptr + pos, mask=seg, other=0).to(tl.int64)
            m = seg[:, None] & cm[None, :]
            z2 = tl.load(z2_ptr + e[:, None] * NG + rc[None, :], mask=m, other=0.0)
            emask = tl.load(mask_ptr + e, mask=seg, other=0.0)
            h2 = activation(z2, ACT) * idt[None, :]
            if RESNET_MULT > 0:
                # concat[h1, h1] (doubling) and identity share ``c mod H1``.
                h1d = tl.load(
                    h1_ptr + e[:, None] * H1 + (rc % H1)[None, :], mask=m, other=0.0
                )
                h2 = h2 + h1d
            if GATED:
                idxe = tl.load(idx_ptr + e, mask=seg, other=0)
                ggt = tl.load(
                    tt_ptr + idxe[:, None] * NG + rc[None, :], mask=m, other=0.0
                )
                swe = tl.load(sw_ptr + e, mask=seg, other=0.0)
                h2 = h2 * (1.0 + ggt * swe[:, None])
            gg = h2 * emask[:, None]
            base = e * 4
            s = tl.load(rr_ptr + base + 0, mask=seg, other=0.0)
            rx = tl.load(rr_ptr + base + 1, mask=seg, other=0.0)
            ry = tl.load(rr_ptr + base + 2, mask=seg, other=0.0)
            rz = tl.load(rr_ptr + base + 3, mask=seg, other=0.0)
            acc0 += tl.sum(s[:, None] * gg, axis=0)
            acc1 += tl.sum(rx[:, None] * gg, axis=0)
            acc2 += tl.sum(ry[:, None] * gg, axis=0)
            acc3 += tl.sum(rz[:, None] * gg, axis=0)
        ob = node * (4 * NG)
        tl.store(gr_ptr + ob + 0 * NG + rc, acc0, mask=cm)
        tl.store(gr_ptr + ob + 1 * NG + rc, acc1, mask=cm)
        tl.store(gr_ptr + ob + 2 * NG + rc, acc2, mask=cm)
        tl.store(gr_ptr + ob + 3 * NG + rc, acc3, mask=cm)

    @triton.jit
    def _edge_conv_bwd_kernel(
        ggr_ptr,  # (N, 4, NG) upstream gradient of the moments
        z2_ptr,
        h1_ptr,
        idt_ptr,
        tt_ptr,  # (P, NG) type-pair embedding table (strip gate)
        idx_ptr,  # (E,) per-edge type-pair row index (strip gate)
        sw_ptr,  # (E,) per-edge smooth cutoff (strip gate)
        rr_ptr,
        dst_ptr,
        mask_ptr,
        dz2_ptr,  # (E, NG)
        dh1_ptr,  # (E, H1); written only when RESNET_MULT > 0
        dsw_ptr,  # (E,); the switch gradient, nonzero only when GATED
        drr_ptr,  # (E, 4)
        n_edge,
        NG: tl.constexpr,
        NGP: tl.constexpr,
        H1: tl.constexpr,
        H1P: tl.constexpr,
        RESNET_MULT: tl.constexpr,
        ACT: tl.constexpr,
        GATED: tl.constexpr,
        BE: tl.constexpr,
    ):
        """Backward of the edge scatter for a block of ``BE`` edges.

        The upstream moment gradient is gathered from ``grad_gr[dst]`` (a per-row
        indirect load, no atomics); ``gg`` is recomputed in registers. The
        residual gradient folds the ``ng // H1`` residual copies -- by a reshape
        reduction when ``H1`` is a power of two, or by recomputing the two halves
        at columns ``rj`` and ``rj + H1`` when ``H1`` is padded (the padded
        reshape would split at ``H1P`` rather than the true ``H1``).

        When ``GATED`` (strip), ``gg = h2 * fac`` with ``fac = 1 + tt[idx] * sw``:
        the gate multiplies ``grad_h2`` (hence ``dz2`` and the residual fold) and
        contributes the switch gradient ``d L / d sw = sum_c dgg * h2 * tt[idx]``.
        The padded-doubling fold gathers the gate for the two halves separately.
        """
        pid = tl.program_id(0)
        e = (pid * BE + tl.arange(0, BE)).to(tl.int64)
        valid = e < n_edge
        rc = tl.arange(0, NGP)
        cm = rc < NG
        m = valid[:, None] & cm[None, :]
        idt = tl.load(idt_ptr + rc, mask=cm, other=0.0)
        z2 = tl.load(z2_ptr + e[:, None] * NG + rc[None, :], mask=m, other=0.0)
        emask = tl.load(mask_ptr + e, mask=valid, other=0.0)
        a, ad = activation_grad(z2, ACT)
        h2 = a * idt[None, :]
        if RESNET_MULT > 0:
            h1d = tl.load(
                h1_ptr + e[:, None] * H1 + (rc % H1)[None, :], mask=m, other=0.0
            )
            h2 = h2 + h1d
        if GATED:
            idxe = tl.load(idx_ptr + e, mask=valid, other=0)
            ggt = tl.load(tt_ptr + idxe[:, None] * NG + rc[None, :], mask=m, other=0.0)
            swe = tl.load(sw_ptr + e, mask=valid, other=0.0)
            fac = 1.0 + ggt * swe[:, None]
        else:
            fac = 1.0 + 0.0 * h2
        gg = h2 * fac * emask[:, None]
        d = tl.load(dst_ptr + e, mask=valid, other=0)
        ob = d[:, None] * (4 * NG) + rc[None, :]
        g0 = tl.load(ggr_ptr + ob + 0 * NG, mask=m, other=0.0)
        g1 = tl.load(ggr_ptr + ob + 1 * NG, mask=m, other=0.0)
        g2 = tl.load(ggr_ptr + ob + 2 * NG, mask=m, other=0.0)
        g3 = tl.load(ggr_ptr + ob + 3 * NG, mask=m, other=0.0)
        base = e * 4
        s = tl.load(rr_ptr + base + 0, mask=valid, other=0.0)
        rx = tl.load(rr_ptr + base + 1, mask=valid, other=0.0)
        ry = tl.load(rr_ptr + base + 2, mask=valid, other=0.0)
        rz = tl.load(rr_ptr + base + 3, mask=valid, other=0.0)
        # d gg = sum_k rr[k] * grad_gr[dst, k]
        dgg = s[:, None] * g0 + rx[:, None] * g1 + ry[:, None] * g2 + rz[:, None] * g3
        grad_h2 = dgg * fac * emask[:, None]
        tl.store(
            dz2_ptr + e[:, None] * NG + rc[None, :], grad_h2 * idt[None, :] * ad, mask=m
        )
        # grad_rr[k] = sum_c gg[c] * grad_gr[dst, k, c]
        tl.store(drr_ptr + base + 0, tl.sum(gg * g0, axis=1), mask=valid)
        tl.store(drr_ptr + base + 1, tl.sum(gg * g1, axis=1), mask=valid)
        tl.store(drr_ptr + base + 2, tl.sum(gg * g2, axis=1), mask=valid)
        tl.store(drr_ptr + base + 3, tl.sum(gg * g3, axis=1), mask=valid)
        if GATED:
            # d L / d sw = sum_c dgg * (d gg / d sw) = sum_c dgg * h2 * tt[idx].
            tl.store(
                dsw_ptr + e,
                tl.sum(dgg * h2 * ggt, axis=1) * emask,
                mask=valid,
            )
        if RESNET_MULT == 1:
            # Identity residual: grad_h1 equals grad_h2 (ng == H1).
            tl.store(dh1_ptr + e[:, None] * H1 + rc[None, :], grad_h2, mask=m)
        elif RESNET_MULT == 2 and H1P == H1:
            grad_h1 = tl.sum(tl.reshape(grad_h2, (BE, 2, H1)), axis=1)
            hj = tl.arange(0, H1)
            tl.store(
                dh1_ptr + e[:, None] * H1 + hj[None, :],
                grad_h1,
                mask=valid[:, None],
            )
        elif RESNET_MULT == 2:
            # Padded doubling: recompute the two halves at rj and rj + H1.
            rj = tl.arange(0, H1P)
            hm = valid[:, None] & (rj < H1)[None, :]
            olo = d[:, None] * (4 * NG) + rj[None, :]
            ohi = olo + H1
            g0lo = tl.load(ggr_ptr + olo + 0 * NG, mask=hm, other=0.0)
            g1lo = tl.load(ggr_ptr + olo + 1 * NG, mask=hm, other=0.0)
            g2lo = tl.load(ggr_ptr + olo + 2 * NG, mask=hm, other=0.0)
            g3lo = tl.load(ggr_ptr + olo + 3 * NG, mask=hm, other=0.0)
            g0hi = tl.load(ggr_ptr + ohi + 0 * NG, mask=hm, other=0.0)
            g1hi = tl.load(ggr_ptr + ohi + 1 * NG, mask=hm, other=0.0)
            g2hi = tl.load(ggr_ptr + ohi + 2 * NG, mask=hm, other=0.0)
            g3hi = tl.load(ggr_ptr + ohi + 3 * NG, mask=hm, other=0.0)
            dgg_lo = (
                s[:, None] * g0lo
                + rx[:, None] * g1lo
                + ry[:, None] * g2lo
                + rz[:, None] * g3lo
            )
            dgg_hi = (
                s[:, None] * g0hi
                + rx[:, None] * g1hi
                + ry[:, None] * g2hi
                + rz[:, None] * g3hi
            )
            if GATED:
                # The gate factor differs per half; gather both columns.
                ggt_lo = tl.load(
                    tt_ptr + idxe[:, None] * NG + rj[None, :], mask=hm, other=0.0
                )
                ggt_hi = tl.load(
                    tt_ptr + idxe[:, None] * NG + (rj + H1)[None, :], mask=hm, other=0.0
                )
                grad_h1 = (
                    dgg_lo * (1.0 + ggt_lo * swe[:, None])
                    + dgg_hi * (1.0 + ggt_hi * swe[:, None])
                ) * emask[:, None]
            else:
                grad_h1 = (dgg_lo + dgg_hi) * emask[:, None]
            tl.store(dh1_ptr + e[:, None] * H1 + rj[None, :], grad_h1, mask=hm)


# ======================================================================
# Dispatch, operator registration and public API
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float32, torch.float64)
    )


def _edge_conv_fwd_impl(
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    dst: Tensor,
    edge_mask: Tensor,
    n_node: int,
    resnet_mult: int,
    act: int,
    gated: int,
    block_e: int,
    num_warps: int,
) -> Tensor:
    if not _use_triton(z2):
        return _edge_conv_reference(
            z2,
            h1,
            idt,
            tt,
            idx,
            sw,
            rr,
            dst,
            edge_mask,
            n_node,
            resnet_mult,
            act,
            gated,
        )
    _, ng = z2.shape
    ngp = triton.next_power_of_2(ng)
    # Destination-sorted CSR topology, built inside the op (integer, gradient-
    # free): ``order`` groups edges by center, ``segptr`` gives each node's
    # segment offsets. The node-parallel reduction is contention-free -- no
    # atomics -- unlike the per-edge scatter it replaces.
    order = torch.argsort(dst)
    boundaries = torch.arange(n_node + 1, device=dst.device, dtype=dst.dtype)
    segptr = torch.searchsorted(dst.index_select(0, order), boundaries).to(torch.int64)
    gr = torch.empty((n_node, 4, ng), dtype=z2.dtype, device=z2.device)
    wrap_triton(_edge_conv_fwd_kernel)[(n_node,)](
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        edge_mask.to(z2.dtype),
        order,
        segptr,
        gr,
        NG=ng,
        NGP=ngp,
        H1=h1.shape[-1],
        RESNET_MULT=resnet_mult,
        ACT=act,
        GATED=gated,
        BE=block_e,
        num_warps=num_warps,
    )
    return gr


def _edge_conv_bwd_impl(
    grad_gr: Tensor,
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    dst: Tensor,
    edge_mask: Tensor,
    n_node: int,
    resnet_mult: int,
    act: int,
    gated: int,
    block_e: int,
    num_warps: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not _use_triton(z2):
        return _edge_conv_reference_backward(
            grad_gr,
            z2,
            h1,
            idt,
            tt,
            idx,
            sw,
            rr,
            dst,
            edge_mask,
            n_node,
            resnet_mult,
            act,
            gated,
        )
    n_edge, ng = z2.shape
    ngp = triton.next_power_of_2(ng)
    dz2 = torch.empty_like(z2)
    dh1 = torch.empty_like(h1) if resnet_mult > 0 else torch.zeros_like(h1)
    dsw = torch.empty_like(sw) if gated else torch.zeros_like(sw)
    drr = torch.empty_like(rr)
    grid = (triton.cdiv(n_edge, block_e),)
    wrap_triton(_edge_conv_bwd_kernel)[grid](
        grad_gr.contiguous(),
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        dst,
        edge_mask.to(z2.dtype),
        dz2,
        dh1,
        dsw,
        drr,
        n_edge,
        NG=ng,
        NGP=ngp,
        H1=h1.shape[-1],
        H1P=triton.next_power_of_2(h1.shape[-1]),
        RESNET_MULT=resnet_mult,
        ACT=act,
        GATED=gated,
        BE=block_e,
        num_warps=num_warps,
    )
    return dz2, dh1, dsw, drr


_edge_conv_op = triton_op("dpa1_triton::edge_conv", mutates_args=())(
    _edge_conv_fwd_impl
)
_edge_conv_bwd_op = triton_op("dpa1_triton::edge_conv_bwd", mutates_args=())(
    _edge_conv_bwd_impl
)


@_edge_conv_op.register_fake
def _(
    z2,
    h1,
    idt,
    tt,
    idx,
    sw,
    rr,
    dst,
    edge_mask,
    n_node,
    resnet_mult,
    act,
    gated,
    block_e,
    num_warps,
):
    return z2.new_empty((n_node, 4, z2.shape[1]))


@_edge_conv_bwd_op.register_fake
def _(
    grad_gr,
    z2,
    h1,
    idt,
    tt,
    idx,
    sw,
    rr,
    dst,
    edge_mask,
    n_node,
    resnet_mult,
    act,
    gated,
    block_e,
    num_warps,
):
    return (
        torch.empty_like(z2),
        torch.empty_like(h1),
        torch.empty_like(sw),
        torch.empty_like(rr),
    )


def _edge_conv_setup_context(ctx, inputs, output):
    (
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        dst,
        edge_mask,
        n_node,
        resnet_mult,
        act,
        gated,
        block_e,
        num_warps,
    ) = inputs
    ctx.save_for_backward(z2, h1, idt, tt, idx, sw, rr, dst, edge_mask)
    ctx.n_node = n_node
    ctx.resnet_mult = resnet_mult
    ctx.act = act
    ctx.gated = gated
    ctx.block_e = block_e
    ctx.num_warps = num_warps


def _edge_conv_backward(ctx, grad_gr):
    z2, h1, idt, tt, idx, sw, rr, dst, edge_mask = ctx.saved_tensors
    grad_z2, grad_h1, grad_sw, grad_rr = _edge_conv_bwd_op(
        grad_gr.contiguous(),
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        dst,
        edge_mask,
        ctx.n_node,
        ctx.resnet_mult,
        ctx.act,
        ctx.gated,
        ctx.block_e,
        ctx.num_warps,
    )
    # ``idt`` / ``tt`` / ``idx`` / ``dst`` / ``edge_mask`` do not depend on
    # coordinates; ``n_node``, ``resnet_mult``, ``act``, ``gated`` and the launch
    # config are static.
    return (
        grad_z2,
        grad_h1,
        None,
        None,
        None,
        grad_sw,
        grad_rr,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_edge_conv_op.register_autograd(
    _edge_conv_backward, setup_context=_edge_conv_setup_context
)


def concat_gate_placeholders(z2: Tensor, ng: int) -> tuple[Tensor, Tensor, Tensor]:
    """Unused ``(tt, idx, sw)`` gate inputs for the concat (``gated == 0``) call.

    Concat carries the type feature through the embedding input, so the strip
    type-pair gate is inactive and :func:`edge_conv` skips these tensors. They
    still fill the operator signature; minimal (single-element) placeholders
    suffice and keep the traced graph lean.

    Parameters
    ----------
    z2 : Tensor
        A tensor sharing the target device and floating dtype.
    ng : int
        Embedding channel width (the ``tt`` table's column count).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        The placeholder ``(tt, idx, sw)``: ``tt`` (1, ng), ``idx`` (1,) int64,
        ``sw`` (1,), matching the per-edge switch layout.
    """
    tt = z2.new_zeros(1, ng)
    idx = torch.zeros(1, dtype=torch.long, device=z2.device)
    sw = z2.new_ones(1)
    return tt, idx, sw


def edge_conv(
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    dst: Tensor,
    edge_mask: Tensor,
    n_node: int,
    resnet_mult: int,
    act: int,
    gated: int,
) -> Tensor:
    """Fused graph-native ``se_atten`` environment convolution (attn-free).

    Parameters
    ----------
    z2 : Tensor
        Last embedding pre-activation with shape (E, ng), where ``E`` is the
        edge count.
    h1 : Tensor
        Penultimate embedding activation with shape (E, H1); read only when
        ``resnet_mult > 0``, where ``ng == resnet_mult * H1``.
    idt : Tensor
        Last-layer per-channel timestep with shape (ng,); ones without
        ``resnet_dt``.
    tt : Tensor
        Type-pair embedding table with shape (P, ng); gathered per edge by
        ``idx`` for the strip gate. A placeholder when ``gated == 0``.
    idx : Tensor
        Per-edge row index into ``tt`` with shape (E,) (strip). One-side uses the
        neighbor type; two-side folds the ``(center, neighbor)`` pair. A
        placeholder when ``gated == 0``.
    sw : Tensor
        Per-edge smooth cutoff with shape (E,) (strip); the switch multiplying
        the gate, or ones when the descriptor is non-smooth. A placeholder when
        ``gated == 0``.
    rr : Tensor
        Per-edge environment matrix ``(s, s*x/r, s*y/r, s*z/r)`` with shape
        (E, 4).
    dst : Tensor
        Center-node index of each edge with shape (E,); the ``segment_sum``
        segment id (arbitrary order).
    edge_mask : Tensor
        Valid-edge mask with shape (E,); padding edges are dropped from the
        reduction (``edge_env_mat`` leaves them nonzero).
    n_node : int
        Number of nodes ``N`` (the segment count).
    resnet_mult : int
        Residual structure of the last layer: ``2`` (width doubling), ``1``
        (identity), or ``0`` (no residual).
    act : int
        Last-layer activation code from :data:`.activation.ACT_CODES`: ``0`` for
        ``tanh``, ``1`` for ``silu``.
    gated : int
        Tebd-input mode: ``1`` applies the strip type-pair gate
        ``gg = h2 * (1 + tt[idx] * sw)``; ``0`` (concat) leaves ``gg = h2`` (the
        type feature entered the embedding input upstream).

    Returns
    -------
    Tensor
        Unnormalized moments ``gr`` with shape (N, 4, ng), equal to the
        ``segment_sum`` of ``rr[:, k] * gg[:, c]`` over edge centers. The
        ``1 / nnei`` normalization is applied by the descriptor afterwards.

    Notes
    -----
    The launch configuration ``(BLOCK_E, num_warps)`` -- the forward's per-node
    segment-block width and the warp count -- is resolved from the active
    ``DP_TRITON_INFER`` level via :func:`.tile_configs.resolve_edge_config`. The
    operator composes under ``make_fx`` / ``torch.compile`` and exposes an
    inference force gradient through the registered backward, so it is baked into
    the pt_expt graph-form ``.pt2`` export.
    """
    block_e, num_warps = resolve_edge_config(
        int(z2.shape[-1]), int(h1.shape[-1]), triton_infer_level()
    )
    return _edge_conv_op(
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        dst,
        edge_mask,
        n_node,
        resnet_mult,
        act,
        gated,
        block_e,
        num_warps,
    )


def _autotune_edge(model: torch.nn.Module, level: int, device: torch.device) -> None:
    """Sweep the ``edge_conv`` launch table for a model about to be frozen.

    Collects the ``(ng, H1)`` shape key of every attention-free ``se_atten``
    descriptor in ``model`` (the graph lower's ``edge_conv`` users, in either
    concat or strip tebd-input mode) and sweeps the keys the built-in /
    freeze-time tables do not yet cover on the target GPU, registering the
    winners so the ``resolve_edge_config`` lookups made while tracing bake tuned
    launches into the graph-form ``.pt2``. Keys already covered cost nothing.
    """
    from deepmd.kernels.triton.dpa1.sweep_tile_configs import (
        sweep_edge,
    )

    device_name = torch.cuda.get_device_name(device)
    keys: set[tuple[int, int]] = set()
    for module in model.modules():
        eligible = getattr(module, "_fused_eligible", None)
        se = getattr(module, "se_atten", None)
        if not (callable(eligible) and eligible("triton")):
            continue
        if se is None or se.tebd_input_mode not in ("concat", "strip"):
            continue
        weight = se.embeddings[0].layers[-1].w
        keys.add((int(weight.shape[1]), int(weight.shape[0])))
    tuned: dict[tuple[int, int], tuple[int, int]] = {}
    for ng, h1 in sorted(keys):
        if has_edge_config(ng, h1):
            continue
        config = sweep_edge(ng, h1, device=device)
        register_edge_config(device_name, ng, h1, config)
        tuned[(ng, h1)] = config
    if tuned:
        log.info("DPA1 edge_conv: tuned launch configs %s on %s.", tuned, device_name)
    else:
        log.info(
            "DPA1 edge_conv: launch table already covers this checkpoint's shapes "
            "on %s; no tuning needed.",
            device_name,
        )


if TRITON_AVAILABLE:
    register_autotuner("dpa1_edge_conv", _autotune_edge)
