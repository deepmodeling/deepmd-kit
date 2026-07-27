# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Fused environment convolution for the DPA1 (``se_atten``) descriptor.

For the attention-free (``attn_layer == 0``), strip-embedding ``se_atten`` path
the per-edge descriptor tail is

    ``h2   = act(z2) * idt + resnet(h1)``    (last embedding layer + resnet)
    ``gg   = h2 * (1 + tt[idx] * sw)``        (type-pair gate + smooth cutoff)
    ``xyz[k, c] = sum_j rr[j, k] * gg[j, c]`` (moment accumulation)

where ``act`` is the last layer's activation (``tanh`` or ``silu``), ``z2`` is
the last embedding pre-activation ``h1 @ W2 + b2``, ``h1`` is the
penultimate activation feeding the resnet, ``idt`` is the last layer's per-
channel timestep (all ones when ``resnet_dt`` is off), ``tt`` is the type-pair
embedding table with per-edge row index ``idx``, ``sw`` is the smooth radial
cutoff, and ``rr`` is the ``(s, s*x/r, s*y/r, s*z/r)`` environment matrix. The
``1 / nnei`` normalization of the moment is applied by the descriptor after this
operator (matching the eager and tabulated paths), so the operator returns the
unnormalized moment ``rr^T @ gg``.

The last-layer resnet takes one of three shapes, selected by ``resnet_mult``
(``= ng // H1`` when the layer adds a residual, else ``0``):

- ``resnet_mult == 2`` -- width doubling, ``resnet(h1)[c] = concat[h1, h1][c]``.
- ``resnet_mult == 1`` -- identity, ``resnet(h1) = h1`` (``ng == H1``).
- ``resnet_mult == 0`` -- no residual; ``h1`` does not enter the activation.

The doubling and identity cases share one addressing rule ``h1[c mod H1]`` and
one backward fold (summing the ``ng // H1`` residual copies), so a single kernel
covers both; the no-residual case skips the ``h1`` read.

The eager path materializes three ``(E, ng)`` tensors (``h2``, the gathered type
feature, and ``gg``) and then runs a batched ``rr^T @ gg`` matmul whose ``M = 4``
contraction uses cuBLAS poorly. This operator fuses the whole tail into one
node-parallel kernel: each program owns one node, streams its neighbors, forms
``gg`` in registers (never materializing any ``(E, ng)`` tensor, and gathering
the type feature inline) and accumulates the four moment rows. The two embedding
GEMMs (``h0 @ W1`` and ``h1 @ W2``) stay on cuBLAS -- in the fp32 regime this
descriptor runs in, Triton ``tl.dot`` has no tensor-core path and cannot beat
cuBLAS, so only the memory-bound, non-GEMM tail is fused. The trailing
``ng x ng x 4`` Gram contraction that forms the final descriptor is likewise
left on cuBLAS.

Design notes and pitfalls
-------------------------
- **Inlined activations (``tanh`` / ``silu``).** The last-layer activation is
  inlined for both value and derivative, selected by the ``act`` code
  (:data:`.activation.ACT_CODES`). Only these activations are eligible; any other
  keeps the dense reference path (the descriptor gates on the activation name).
  Both are smooth, so the inference force stays differentiable.
- **Input-precision accumulation.** The moment is accumulated in the input
  dtype -- fp32 for the eager DPA1 path, fp64 for the float64 pt_expt / export
  path -- and no ``tl.dot`` is used.
- **Arbitrary channel widths via padding.** ``tl.arange`` requires a
  power-of-two bound, so the channel axis is carried at the padded width
  ``next_pow2(ng)`` and the ``next_pow2(ng) - ng`` surplus lanes are masked to
  zero on every load and skipped on every store. Padding to the next power of
  two (rather than tiling the channel axis) keeps the kernel single-pass and
  is uniformly faster than the eager path even at the worst padding ratio,
  because the extra lanes are pure bandwidth with no cross-lane dependence. The
  backward residual fold additionally depends on the true (unpadded) ``H1``;
  see :func:`_se_conv_bwd_kernel`.
- **Backward register footprint.** The backward holds a ``(BLOCK_N, ng)`` block
  live; an oversized ``BLOCK_N`` spills and collapses throughput far more
  sharply than in the forward. Configurations come from
  :func:`.tile_configs.resolve_conv_config`, whose default is deliberately
  small.
- **Inference-only autograd.** The registered backward returns gradients for
  the coordinate-bearing inputs (``z2``, ``h1``, ``sw``, ``rr``) that carry the
  force; the timestep, type table and index do not depend on coordinates.
  Training keeps the dense reference path (the gate is inference-only), so their
  gradients are never required here.
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
    ACT_CODES,
    TRITON_AVAILABLE,
)
from deepmd.kernels.triton.dpa1.gemm_fp16x3 import (
    embed_last_gemm,
)
from deepmd.kernels.triton.dpa1.tile_configs import (
    has_conv_config,
    register_conv_config,
    resolve_conv_config,
)
from deepmd.kernels.utils import (
    triton_infer_level,
)

log = logging.getLogger(__name__)

__all__ = [
    "concat_gate_placeholders",
    "se_atten_conv",
    "se_conv",
]


# ======================================================================
# Eager reference / fallback implementation
# ======================================================================
def _se_conv_reference(
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    resnet_mult: int,
    act: int,
    gated: int,
) -> Tensor:
    """Eager ground truth for the fused environment convolution.

    ``gated`` selects the tebd-input mode: ``1`` (strip) applies the type-pair
    gate; ``0`` (concat) skips it (the type feature enters upstream through the
    embedding input), so ``tt`` / ``idx`` / ``sw`` are ignored.
    """
    nfnl, nnei, ng = z2.shape
    activated = torch.tanh(z2) if act == 0 else F.silu(z2)
    h2 = activated * idt
    if resnet_mult == 2:
        h2 = h2 + torch.cat([h1, h1], dim=-1)
    elif resnet_mult == 1:
        h2 = h2 + h1
    if gated:
        gg_t = tt.index_select(0, idx.reshape(-1)).reshape(nfnl, nnei, ng)
        gg = h2 * (1.0 + gg_t * sw.unsqueeze(-1))
    else:
        gg = h2
    return torch.matmul(rr.transpose(1, 2), gg)


def _se_conv_reference_backward(
    grad_xyz: Tensor,
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    resnet_mult: int,
    act: int,
    gated: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Closed-form gradient of :func:`_se_conv_reference` w.r.t. the coordinate-
    bearing inputs ``(z2, h1, sw, rr)``.

    A closed form (rather than a nested ``torch.autograd.grad``) is used so the
    fallback composes under ``make_fx`` / ``torch.export``: the tracer runs this
    body with grad tracking disabled, where an inner autograd call would find no
    graph. The expressions mirror the Triton backward kernel. ``gated == 0``
    (concat) drops the type gate, so ``fac == 1`` and ``grad_sw == 0``.
    """
    nfnl, nnei, ng = z2.shape
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
    if gated:
        gg_t = tt.index_select(0, idx.reshape(-1)).reshape(nfnl, nnei, ng)
        fac = 1.0 + gg_t * sw.unsqueeze(-1)  # d gg / d h2
    else:
        fac = torch.ones_like(h2)
    gg = h2 * fac
    # d(xyz = rr^T @ gg): grad_gg = rr @ grad_xyz, grad_rr = gg @ grad_xyz^T.
    grad_gg = torch.matmul(rr, grad_xyz)
    grad_rr = torch.matmul(gg, grad_xyz.transpose(1, 2))
    grad_h2 = grad_gg * fac
    grad_z2 = grad_h2 * idt * act_grad
    # fac = 1 + gg_t * sw, so d gg / d sw = h2 * gg_t (zero without the gate).
    if gated:
        grad_sw = (grad_gg * h2 * gg_t).sum(dim=-1)
    else:
        grad_sw = torch.zeros_like(sw)
    if resnet_mult == 2:
        hh = h1.shape[-1]
        grad_h1 = grad_h2[..., :hh] + grad_h2[..., hh : 2 * hh]
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
    def _se_conv_fwd_kernel(
        z2_ptr,  # (N, NNEI, NG) last embedding pre-activation
        h1_ptr,  # (N, NNEI, H1) penultimate activation (resnet source)
        idt_ptr,  # (NG,) last-layer timestep (ones without resnet_dt)
        tt_ptr,  # (P, NG) type-pair embedding table
        idx_ptr,  # (N * NNEI,) per-edge type-pair row index
        sw_ptr,  # (N, NNEI) smooth radial cutoff
        rr_ptr,  # (N, NNEI, 4) environment matrix
        out_ptr,  # (N, 4, NG) accumulated moments
        NNEI,
        H1: tl.constexpr,
        NG: tl.constexpr,
        NGP: tl.constexpr,
        RESNET_MULT: tl.constexpr,
        ACT: tl.constexpr,
        GATED: tl.constexpr,
        BN: tl.constexpr,
    ):
        """Accumulate the four unnormalized moment rows over a node's neighbors.

        ``gg`` is formed in registers per neighbor block and never written to
        global memory; the type feature is gathered inline through ``idx``. The
        channel axis is carried at the padded width ``NGP = next_pow2(NG)`` (the
        ``tl.arange`` bound must be a power of two); the ``NGP - NG`` padding
        lanes are masked to zero on every load and skipped on the final store,
        so they contribute nothing to the moment reduction.

        ``GATED`` selects the tebd-input mode: ``1`` (strip) applies the
        type-pair gate ``gg = h2 * (1 + tt[idx] * sw)``; ``0`` (concat) skips it
        (``gg = h2``), since concat feeds the type feature through the embedding
        input instead of a multiplicative gate.
        """
        node = tl.program_id(0)
        rc = tl.arange(0, NGP)
        # The channel mask is only material when the width is padded (NGP > NG);
        # for a power-of-two width it is compile-time all-true. It is elided from
        # the per-neighbor loads and the ``dz2`` store (the bandwidth-bound hot
        # path) so an unpadded launch matches the plain neighbor-masked fast path
        # exactly; the cheap once-per-node loads/stores keep it unconditionally.
        cm = rc < NG
        idt = tl.load(idt_ptr + rc, mask=cm, other=0.0)
        # Accumulate in the input precision so the kernel serves both fp32
        # (the eager DPA1 path) and fp64 (the float64 pt_expt / export path).
        acc_ty = z2_ptr.dtype.element_ty
        acc0 = tl.zeros((NGP,), dtype=acc_ty)
        acc1 = tl.zeros((NGP,), dtype=acc_ty)
        acc2 = tl.zeros((NGP,), dtype=acc_ty)
        acc3 = tl.zeros((NGP,), dtype=acc_ty)
        for n0 in range(0, NNEI, BN):
            offs = n0 + tl.arange(0, BN)
            nmask = offs < NNEI
            m = (nmask[:, None] & cm[None, :]) if NGP != NG else nmask[:, None]
            e = node * NNEI + offs
            z2 = tl.load(z2_ptr + e[:, None] * NG + rc[None, :], mask=m, other=0.0)
            base = e * 4
            s = tl.load(rr_ptr + base + 0, mask=nmask, other=0.0)
            rx = tl.load(rr_ptr + base + 1, mask=nmask, other=0.0)
            ry = tl.load(rr_ptr + base + 2, mask=nmask, other=0.0)
            rz = tl.load(rr_ptr + base + 3, mask=nmask, other=0.0)
            h2 = activation(z2, ACT) * idt[None, :]
            if RESNET_MULT > 0:
                # concat[h1, h1] (doubling) and identity share one addressing
                # rule; ``c mod H1`` reads the residual source for either width.
                h1d = tl.load(
                    h1_ptr + e[:, None] * H1 + (rc % H1)[None, :], mask=m, other=0.0
                )
                h2 = h2 + h1d
            if GATED:
                idx = tl.load(idx_ptr + e, mask=nmask, other=0)
                ggt = tl.load(
                    tt_ptr + idx[:, None] * NG + rc[None, :], mask=m, other=0.0
                )
                sw = tl.load(sw_ptr + e, mask=nmask, other=0.0)
                gg = h2 * (1.0 + ggt * sw[:, None])
            else:
                gg = h2
            acc0 += tl.sum(s[:, None] * gg, axis=0)
            acc1 += tl.sum(rx[:, None] * gg, axis=0)
            acc2 += tl.sum(ry[:, None] * gg, axis=0)
            acc3 += tl.sum(rz[:, None] * gg, axis=0)
        ob = node * 4 * NG
        tl.store(out_ptr + ob + 0 * NG + rc, acc0, mask=cm)
        tl.store(out_ptr + ob + 1 * NG + rc, acc1, mask=cm)
        tl.store(out_ptr + ob + 2 * NG + rc, acc2, mask=cm)
        tl.store(out_ptr + ob + 3 * NG + rc, acc3, mask=cm)

    @triton.jit
    def _se_conv_bwd_kernel(
        gout_ptr,  # (N, 4, NG) upstream gradient of the moments
        z2_ptr,
        h1_ptr,
        idt_ptr,
        tt_ptr,
        idx_ptr,
        sw_ptr,
        rr_ptr,
        dz2_ptr,  # (N, NNEI, NG)
        dh1_ptr,  # (N, NNEI, H1); written only when RESNET_MULT > 0
        dsw_ptr,  # (N, NNEI)
        drr_ptr,  # (N, NNEI, 4)
        NNEI,
        H1: tl.constexpr,
        NG: tl.constexpr,
        NGP: tl.constexpr,
        H1P: tl.constexpr,
        RESNET_MULT: tl.constexpr,
        ACT: tl.constexpr,
        GATED: tl.constexpr,
        BN: tl.constexpr,
    ):
        """Backward of the moment accumulation for one node.

        The upstream moment gradient is loaded once per node and broadcast over
        the neighbor stream; ``gg`` is recomputed in registers (mirroring the
        forward), and the channel axis is padded to ``NGP`` with masked loads.

        The residual gradient ``grad_h1`` folds the ``ng // H1`` residual copies.
        When ``H1`` is a power of two the fold is the free reduction of the
        ``(BLOCK_N, ng / H1, H1)`` view. When ``H1`` is padded (``H1P != H1``)
        that view would split the copies at ``H1P`` rather than the true ``H1``,
        so the doubling fold is instead evaluated directly as
        ``grad_h2[:, :H1] + grad_h2[:, H1:2*H1]`` from the two upstream halves.
        """
        node = tl.program_id(0)
        rc = tl.arange(0, NGP)
        cm = rc < NG
        idt = tl.load(idt_ptr + rc, mask=cm, other=0.0)
        gb = node * (4 * NG)
        g0 = tl.load(gout_ptr + gb + 0 * NG + rc, mask=cm, other=0.0)
        g1 = tl.load(gout_ptr + gb + 1 * NG + rc, mask=cm, other=0.0)
        g2 = tl.load(gout_ptr + gb + 2 * NG + rc, mask=cm, other=0.0)
        g3 = tl.load(gout_ptr + gb + 3 * NG + rc, mask=cm, other=0.0)
        # Doubling with padded H1 needs the moment gradient at both residual
        # halves; hoist the per-node loads at columns ``rj`` and ``rj + H1``.
        if RESNET_MULT == 2 and H1P != H1:
            rj = tl.arange(0, H1P)
            hm = rj < H1
            g0lo = tl.load(gout_ptr + gb + 0 * NG + rj, mask=hm, other=0.0)
            g1lo = tl.load(gout_ptr + gb + 1 * NG + rj, mask=hm, other=0.0)
            g2lo = tl.load(gout_ptr + gb + 2 * NG + rj, mask=hm, other=0.0)
            g3lo = tl.load(gout_ptr + gb + 3 * NG + rj, mask=hm, other=0.0)
            g0hi = tl.load(gout_ptr + gb + 0 * NG + rj + H1, mask=hm, other=0.0)
            g1hi = tl.load(gout_ptr + gb + 1 * NG + rj + H1, mask=hm, other=0.0)
            g2hi = tl.load(gout_ptr + gb + 2 * NG + rj + H1, mask=hm, other=0.0)
            g3hi = tl.load(gout_ptr + gb + 3 * NG + rj + H1, mask=hm, other=0.0)
        for n0 in range(0, NNEI, BN):
            offs = n0 + tl.arange(0, BN)
            nmask = offs < NNEI
            m = (nmask[:, None] & cm[None, :]) if NGP != NG else nmask[:, None]
            e = node * NNEI + offs
            ec = e[:, None] * NG + rc[None, :]
            z2 = tl.load(z2_ptr + ec, mask=m, other=0.0)
            base = e * 4
            s = tl.load(rr_ptr + base + 0, mask=nmask, other=0.0)
            rx = tl.load(rr_ptr + base + 1, mask=nmask, other=0.0)
            ry = tl.load(rr_ptr + base + 2, mask=nmask, other=0.0)
            rz = tl.load(rr_ptr + base + 3, mask=nmask, other=0.0)
            a, ad = activation_grad(z2, ACT)
            h2 = a * idt[None, :]
            if RESNET_MULT > 0:
                h1d = tl.load(
                    h1_ptr + e[:, None] * H1 + (rc % H1)[None, :], mask=m, other=0.0
                )
                h2 = h2 + h1d
            if GATED:
                idx = tl.load(idx_ptr + e, mask=nmask, other=0)
                ggt = tl.load(
                    tt_ptr + idx[:, None] * NG + rc[None, :], mask=m, other=0.0
                )
                sw = tl.load(sw_ptr + e, mask=nmask, other=0.0)
                fac = 1.0 + ggt * sw[:, None]  # d gg / d h2
            else:
                fac = 1.0
            gg = h2 * fac
            dgg = (
                s[:, None] * g0[None, :]
                + rx[:, None] * g1[None, :]
                + ry[:, None] * g2[None, :]
                + rz[:, None] * g3[None, :]
            )
            grad_h2 = dgg * fac
            tl.store(
                dz2_ptr + ec,
                grad_h2 * idt[None, :] * ad,
                mask=m,
            )
            if RESNET_MULT == 1:
                # Identity residual: ``grad_h1`` equals the moment gradient.
                tl.store(dh1_ptr + ec, grad_h2, mask=m)
            elif RESNET_MULT == 2 and H1P == H1:
                grad_h1 = tl.sum(tl.reshape(grad_h2, (BN, 2, H1)), axis=1)
                tl.store(
                    dh1_ptr + e[:, None] * H1 + tl.arange(0, H1)[None, :],
                    grad_h1,
                    mask=nmask[:, None],
                )
            elif RESNET_MULT == 2:
                # Padded doubling: recompute the two residual halves directly.
                hmask = nmask[:, None] & (rj < H1)[None, :]
                dgg_lo = (
                    s[:, None] * g0lo[None, :]
                    + rx[:, None] * g1lo[None, :]
                    + ry[:, None] * g2lo[None, :]
                    + rz[:, None] * g3lo[None, :]
                )
                dgg_hi = (
                    s[:, None] * g0hi[None, :]
                    + rx[:, None] * g1hi[None, :]
                    + ry[:, None] * g2hi[None, :]
                    + rz[:, None] * g3hi[None, :]
                )
                if GATED:
                    ggt_lo = tl.load(
                        tt_ptr + idx[:, None] * NG + rj[None, :], mask=hmask, other=0.0
                    )
                    ggt_hi = tl.load(
                        tt_ptr + idx[:, None] * NG + rj[None, :] + H1,
                        mask=hmask,
                        other=0.0,
                    )
                    grad_h1 = dgg_lo * (1.0 + ggt_lo * sw[:, None]) + dgg_hi * (
                        1.0 + ggt_hi * sw[:, None]
                    )
                else:
                    grad_h1 = dgg_lo + dgg_hi
                tl.store(dh1_ptr + e[:, None] * H1 + rj[None, :], grad_h1, mask=hmask)
            if GATED:
                # sw enters only through the gate; concat has no sw gradient.
                tl.store(dsw_ptr + e, tl.sum(dgg * h2 * ggt, axis=1), mask=nmask)
            tl.store(drr_ptr + base + 0, tl.sum(gg * g0[None, :], axis=1), mask=nmask)
            tl.store(drr_ptr + base + 1, tl.sum(gg * g1[None, :], axis=1), mask=nmask)
            tl.store(drr_ptr + base + 2, tl.sum(gg * g2[None, :], axis=1), mask=nmask)
            tl.store(drr_ptr + base + 3, tl.sum(gg * g3[None, :], axis=1), mask=nmask)


# ======================================================================
# Dispatch, operator registration and public API
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float32, torch.float64)
    )


def _se_conv_fwd_impl(
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    resnet_mult: int,
    act: int,
    gated: int,
    block_n: int,
    num_warps: int,
) -> Tensor:
    if not _use_triton(z2):
        return _se_conv_reference(z2, h1, idt, tt, idx, sw, rr, resnet_mult, act, gated)
    nfnl, nnei, ng = z2.shape
    out = torch.empty((nfnl, 4, ng), dtype=z2.dtype, device=z2.device)
    wrap_triton(_se_conv_fwd_kernel)[(nfnl,)](
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        out,
        nnei,
        H1=h1.shape[-1],
        NG=ng,
        NGP=triton.next_power_of_2(ng),
        RESNET_MULT=resnet_mult,
        ACT=act,
        GATED=gated,
        BN=block_n,
        num_warps=num_warps,
    )
    return out


def _se_conv_bwd_impl(
    grad_xyz: Tensor,
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    resnet_mult: int,
    act: int,
    gated: int,
    block_n: int,
    num_warps: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not _use_triton(z2):
        return _se_conv_reference_backward(
            grad_xyz, z2, h1, idt, tt, idx, sw, rr, resnet_mult, act, gated
        )
    nfnl, nnei, ng = z2.shape
    dz2 = torch.empty_like(z2)
    # ``h1`` enters the activation only through the residual; with no residual
    # its moment gradient is zero (its GEMM gradient is formed outside the op).
    dh1 = torch.empty_like(h1) if resnet_mult > 0 else torch.zeros_like(h1)
    # Concat (gated == 0) uses no sw gate; the kernel skips the dsw store, so
    # pre-zero it here.
    dsw = torch.empty_like(sw) if gated else torch.zeros_like(sw)
    drr = torch.empty_like(rr)
    wrap_triton(_se_conv_bwd_kernel)[(nfnl,)](
        grad_xyz.contiguous(),
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        dz2,
        dh1,
        dsw,
        drr,
        nnei,
        H1=h1.shape[-1],
        NG=ng,
        NGP=triton.next_power_of_2(ng),
        H1P=triton.next_power_of_2(h1.shape[-1]),
        RESNET_MULT=resnet_mult,
        ACT=act,
        GATED=gated,
        BN=block_n,
        num_warps=num_warps,
    )
    return dz2, dh1, dsw, drr


_se_conv_op = triton_op("dpa1_triton::se_conv", mutates_args=())(_se_conv_fwd_impl)
_se_conv_bwd_op = triton_op("dpa1_triton::se_conv_bwd", mutates_args=())(
    _se_conv_bwd_impl
)


@_se_conv_op.register_fake
def _(z2, h1, idt, tt, idx, sw, rr, resnet_mult, act, gated, block_n, num_warps):
    return z2.new_empty((z2.shape[0], 4, z2.shape[2]))


@_se_conv_bwd_op.register_fake
def _(
    grad_xyz, z2, h1, idt, tt, idx, sw, rr, resnet_mult, act, gated, block_n, num_warps
):
    return (
        torch.empty_like(z2),
        torch.empty_like(h1),
        torch.empty_like(sw),
        torch.empty_like(rr),
    )


def _se_conv_setup_context(ctx, inputs, output):
    z2, h1, idt, tt, idx, sw, rr, resnet_mult, act, gated, block_n, num_warps = inputs
    ctx.save_for_backward(z2, h1, idt, tt, idx, sw, rr)
    ctx.resnet_mult = resnet_mult
    ctx.act = act
    ctx.gated = gated
    ctx.block_n = block_n
    ctx.num_warps = num_warps


def _se_conv_backward(ctx, grad_xyz):
    z2, h1, idt, tt, idx, sw, rr = ctx.saved_tensors
    grad_z2, grad_h1, grad_sw, grad_rr = _se_conv_bwd_op(
        grad_xyz.contiguous(),
        z2,
        h1,
        idt,
        tt,
        idx,
        sw,
        rr,
        ctx.resnet_mult,
        ctx.act,
        ctx.gated,
        ctx.block_n,
        ctx.num_warps,
    )
    # ``idt`` / ``tt`` / ``idx`` do not depend on coordinates (no force grad).
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
    )


_se_conv_op.register_autograd(_se_conv_backward, setup_context=_se_conv_setup_context)


def concat_gate_placeholders(z2: Tensor, ng: int) -> tuple[Tensor, Tensor, Tensor]:
    """Unused ``(tt, idx, sw)`` gate inputs for the concat (``gated == 0``) call.

    Concat carries the type feature through the embedding input, so the strip
    type-pair gate is inactive and :func:`se_conv` skips these tensors. They are
    still required to fill the operator signature; minimal (single-element)
    placeholders suffice and keep the traced graph lean.

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
        ``sw`` (1, 1).
    """
    tt = z2.new_zeros(1, ng)
    idx = torch.zeros(1, dtype=torch.long, device=z2.device)
    sw = z2.new_ones(1, 1)
    return tt, idx, sw


def se_conv(
    z2: Tensor,
    h1: Tensor,
    idt: Tensor,
    tt: Tensor,
    idx: Tensor,
    sw: Tensor,
    rr: Tensor,
    resnet_mult: int,
    act: int,
    gated: int,
) -> Tensor:
    """Fused ``se_atten`` environment convolution (attn-free, strip or concat).

    Parameters
    ----------
    z2 : Tensor
        Last embedding pre-activation ``h1 @ W2 + b2`` with shape
        (N, nnei, ng), where ``N = nframes * nloc``.
    h1 : Tensor
        Penultimate embedding activation with shape (N, nnei, H1); read only
        when ``resnet_mult > 0``, where ``ng == resnet_mult * H1``.
    idt : Tensor
        Last-layer per-channel timestep with shape (ng,); all ones when the
        network has no ``resnet_dt``.
    tt : Tensor
        Type-pair embedding table with shape (P, ng).
    idx : Tensor
        Per-edge row index into ``tt`` with shape (N * nnei,), dtype int64.
    sw : Tensor
        Smooth radial cutoff with shape (N, nnei).
    rr : Tensor
        Environment matrix ``(s, s*x/r, s*y/r, s*z/r)`` with shape (N, nnei, 4).
    resnet_mult : int
        Residual structure of the last layer: ``2`` (width doubling), ``1``
        (identity), or ``0`` (no residual).
    act : int
        Last-layer activation code from :data:`.activation.ACT_CODES`: ``0`` for
        ``tanh``, ``1`` for ``silu``.
    gated : int
        Tebd-input mode: ``1`` (strip) applies the type-pair gate
        ``gg = h2 * (1 + tt[idx] * sw)``; ``0`` (concat) skips it (``gg = h2``)
        and ignores ``tt`` / ``idx`` / ``sw``, since concat feeds the type
        feature through the embedding input.

    Returns
    -------
    Tensor
        Unnormalized moments ``xyz`` with shape (N, 4, ng), equal to
        ``rr^T @ gg``. The ``1 / nnei`` normalization is applied by the
        descriptor after this operator.

    Notes
    -----
    The launch configuration is resolved from the active ``DP_TRITON_INFER``
    level via :func:`.tile_configs.resolve_conv_config`. The operator composes
    under ``make_fx`` / ``torch.compile`` and exposes an inference force
    gradient through the registered backward.
    """
    block_n, num_warps = resolve_conv_config(
        int(z2.shape[-1]), int(h1.shape[-1]), triton_infer_level()
    )
    return _se_conv_op(
        z2, h1, idt, tt, idx, sw, rr, resnet_mult, act, gated, block_n, num_warps
    )


def se_atten_conv(
    embedding_net,
    ss: Tensor,
    tt: Tensor | None,
    idx: Tensor | None,
    sw: Tensor | None,
    rr: Tensor,
    gated: int,
) -> Tensor:
    """Fuse the embedding net's final layer, type gate and moment accumulation.

    The embedding net is evaluated through its penultimate layer on the eager
    (cuBLAS) path -- fp32 GEMMs which Triton cannot beat without tensor cores --
    and the final layer's activation (its timestep and residual) is folded into
    :func:`se_conv` together with the type-pair gate (strip only), smooth cutoff
    and the environment moment reduction.

    Parameters
    ----------
    embedding_net : EmbeddingNet
        The strip-mode radial embedding network. Any channel width is supported;
        widths that are not powers of two are handled by masked padding inside
        :func:`se_conv`. The last layer's activation must be one of
        :data:`.activation.ACT_CODES` (``tanh`` or ``silu``); the caller verifies
        this before routing here.
    ss : Tensor
        Embedding-net input with shape (N, nnei, d_in): the radial channel
        (d_in == 1) for strip (``gated``), or the radial-plus-type-embedding
        concatenation for concat (``not gated``).
    tt : Tensor or None
        Type-pair embedding table with shape (P, ng); the strip gate table.
        Ignored (pass ``None``) for concat, whose type feature is already in
        ``ss``.
    idx : Tensor or None
        Per-edge row index into ``tt`` with shape (N * nnei,); ``None`` for
        concat.
    sw : Tensor or None
        Smooth radial cutoff with shape (N, nnei); ``None`` for concat.
    rr : Tensor
        Environment matrix with shape (N, nnei, 4).
    gated : int
        ``1`` (strip) applies the type-pair gate; ``0`` (concat) skips it.

    Returns
    -------
    Tensor
        Unnormalized moments ``xyz`` with shape (N, 4, ng).
    """
    *head, last = embedding_net.layers
    h = ss
    for layer in head:
        h = layer(h)
    z2 = embed_last_gemm(h, last.matrix, last.bias)
    h1_dim, ng = last.matrix.shape
    if last.resnet and ng == 2 * h1_dim:
        resnet_mult = 2
    elif last.resnet and ng == h1_dim:
        resnet_mult = 1
    else:
        resnet_mult = 0
    idt = last.idt if last.idt is not None else z2.new_ones(ng)
    act = ACT_CODES[last.activate_name]
    if not gated:
        tt, idx, sw = concat_gate_placeholders(z2, ng)
    return se_conv(
        z2.contiguous(), h.contiguous(), idt, tt, idx, sw, rr, resnet_mult, act, gated
    )


# ======================================================================
# Freeze-time launch-configuration autotuning (DP_TRITON_INFER >= 2)
# ======================================================================
def _autotune_conv(model: torch.nn.Module, level: int, device: torch.device) -> None:
    """Sweep the fused-convolution launch table for a model about to be frozen.

    Collects the ``(ng, H1)`` shape key of every eligible ``se_atten``
    descriptor in ``model`` and sweeps the keys the built-in / freeze-time
    tables do not yet cover on the target GPU, registering the winners so the
    ``resolve_conv_config`` lookups made while tracing bake tuned launches into
    the exported ``.pt2``. Keys already covered cost nothing.
    """
    from deepmd.kernels.triton.dpa1.sweep_tile_configs import (
        sweep,
    )

    device_name = torch.cuda.get_device_name(device)
    keys: set[tuple[int, int]] = set()
    for module in model.modules():
        eligible = getattr(module, "_fused_eligible", None)
        if not (callable(eligible) and eligible("triton")):
            continue
        weight = module.se_atten.embeddings[0].layers[-1].w
        keys.add((int(weight.shape[1]), int(weight.shape[0])))
    tuned: dict[tuple[int, int], tuple[int, int]] = {}
    for ng, h1 in sorted(keys):
        # The sweep needs a residual last layer (ng in {H1, 2*H1}); other shapes
        # keep the default. Skip keys the tables already cover.
        if has_conv_config(ng, h1) or ng not in (h1, 2 * h1):
            continue
        config = sweep(ng, h1, device=device)
        register_conv_config(device_name, ng, h1, config)
        tuned[(ng, h1)] = config
    if tuned:
        log.info("DPA1 se_conv: tuned launch configs %s on %s.", tuned, device_name)
    else:
        log.info(
            "DPA1 se_conv: launch table already covers this checkpoint's shapes "
            "on %s; no tuning needed.",
            device_name,
        )


if TRITON_AVAILABLE:
    register_autotuner("dpa1_se_conv", _autotune_conv)
