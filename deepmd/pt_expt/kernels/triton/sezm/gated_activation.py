# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
r"""Second order of the gated SO(2) activation.

The SO(2) mixing stack is the only part of the SeZM descriptor that is not
multilinear, so it is the only one whose second derivative cannot be assembled
from its own forward and backward. This module supplies that derivative for the
nonlinear core of one layer; the stack orchestration (the block GEMMs and the
traversal over layers) lives in :mod:`.so2_value_path`.

Operation
---------
Within one layer the pre-activation ``z`` is split into the scalar rows
:math:`s` (the ``l = 0`` block, ``Cf`` channels) and, for every degree
:math:`g = 1 \ldots L`, a triple of value rows sharing one gate group: the
``m = 0`` row and the ``m = \pm 1`` pair. With :math:`G` the gate projection,

.. math::

    q = s\,G, \qquad
    \mathrm{act} = \bigl[\, s\,\sigma(s),\;
        z_{r}\,\sigma(q)_{g(r)} \,\bigr],

where :math:`r` runs over the value rows and :math:`g(r)` is the gate group of
that row. The scalar rows therefore drive every gate, which is what couples the
three rows of a group and makes the second order non-separable.

First order
-----------
Given the output cotangent :math:`\bar u`, writing
:math:`A(s) = \sigma(s)\bigl(1 + s(1-\sigma(s))\bigr)` for the SiLU derivative,

.. math::

    \bar z_s &= \bar u_s\,A(s) + \bigl(\bar q\,\sigma'(q)\bigr) G^{\mathsf T},\\
    \bar z_r &= \bar u_r\,\sigma(q)_{g(r)},\\
    \bar q_g &= \Sigma_g\,\sigma'(q)_g,
        \qquad \Sigma_g = \sum_{r \in g} \bar u_r z_r .

The contraction of the gate-logit gradient back onto the scalar rows is either
folded into this kernel or left to the caller, depending on which is cheaper at
the channel width in play; ``fold_logit`` selects between them.

Second order
------------
Differentiating the first order at incoming cotangents :math:`h_z` (of
:math:`\bar z`) and :math:`h_q` (of :math:`\bar q`) needs the second derivatives

.. math::

    A'(s) = \sigma(s)\bigl(1-\sigma(s)\bigr)
            \bigl[2 + s\bigl(1-2\sigma(s)\bigr)\bigr],
    \qquad
    \sigma''(q) = \sigma'(q)\bigl(1 - 2\sigma(q)\bigr).

With the effective logit cotangent
:math:`\tilde h_q = h_q + h_{z,s} G` (the second term present only when the
contraction was folded in) and :math:`w_g = \tilde h_{q,g}\,\sigma'(q)_g`,

.. math::

    \frac{\partial S}{\partial \bar u_s} &= h_{z,s}\,A(s), &
    \frac{\partial S}{\partial \bar u_r} &= h_{z,r}\,\sigma(q)_{g(r)}
        + w_{g(r)} z_r, \\
    \frac{\partial S}{\partial z_r} &= w_{g(r)}\,\bar u_r, &
    \frac{\partial S}{\partial z_s} &= h_{z,s}\,\bar u_s\,A'(s)
        + \bigl(\partial S/\partial q\bigr) G^{\mathsf T},

with the logit term collecting both routes through the gate,

.. math::

    \frac{\partial S}{\partial q_g}
      = \Bigl(\sum_{r \in g} h_{z,r}\,\bar u_r\Bigr)\sigma'(q)_g
      + \tilde h_{q,g}\,\Sigma_g\,\sigma''(q)_g .

The gate projection's gradient, :math:`s^{\mathsf T}(\partial S/\partial q)`
plus :math:`h_{z,s}^{\mathsf T}\bar q` when the contraction was folded in,
reduces the whole edge axis and is left to cuBLAS; everything above is
elementwise apart from two ``Cf x LG`` projections and is fused into one kernel.
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

from .tile_configs import (
    gated_second_order_config,
)

__all__ = [
    "GATED_ACTIVATION_TRITON_AVAILABLE",
    "gated_activation_second_order",
]

try:
    import triton
    import triton.language as tl

    GATED_ACTIVATION_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    GATED_ACTIVATION_TRITON_AVAILABLE = False


def gated_activation_second_order_reference(
    grad_grad_z: Tensor | None,
    grad_grad_logit: Tensor | None,
    grad: Tensor,
    z: Tensor,
    gw: Tensor,
    lmax: int,
    focus_dim: int,
    fold_logit: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager ground truth for :func:`gated_activation_second_order`.

    Parameters
    ----------
    grad_grad_z : Tensor or None
        Cotangent of the pre-activation gradient, with shape ``(F, E, ROW)``.
    grad_grad_logit : Tensor or None
        Cotangent of the gate-logit gradient, with shape ``(F, E, lmax * Cf)``.
    grad : Tensor
        Output cotangent of the layer, with shape ``(F, E, ROW)``.
    z : Tensor
        Pre-activation of the layer, with shape ``(F, E, ROW)``.
    gw : Tensor
        Gate projection, with shape ``(F, Cf, lmax * Cf)``.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.
    fold_logit : bool
        Whether the caller, rather than this operator, contracts the gate-logit
        gradient back onto the scalar rows.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Gradients with respect to ``(grad, z, gw)``.
    """
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    m0 = (lmax + 1) * focus_dim
    n_focus, n_edge = z.shape[0], z.shape[1]

    def fold(value: Tensor) -> Tensor:
        """Sum the two signed-``m`` halves that share a gate group."""
        return value.view(n_focus, n_edge, 2, -1).sum(2)

    scalar = z[:, :, :focus_dim]
    grad_scalar = grad[:, :, :focus_dim]
    grad_gated0 = grad[:, :, focus_dim:m0]
    grad_gated1 = grad[:, :, m0:]
    z_gated0 = z[:, :, focus_dim:m0]
    z_gated1 = z[:, :, m0:]

    sig = torch.sigmoid(torch.bmm(scalar, gw))
    d_sig = sig * (1.0 - sig)
    dd_sig = d_sig * (1.0 - 2.0 * sig)
    silu_sig = torch.sigmoid(scalar)
    dd_silu = silu_sig * (1.0 - silu_sig) * (2.0 + scalar * (1.0 - 2.0 * silu_sig))

    grad_sig = grad_gated0 * z_gated0 + fold(grad_gated1 * z_gated1)

    zero_logit = torch.zeros_like(sig)
    hz_scalar = (
        grad_grad_z[:, :, :focus_dim]
        if grad_grad_z is not None
        else torch.zeros_like(scalar)
    )
    hz_gated0 = (
        grad_grad_z[:, :, focus_dim:m0] if grad_grad_z is not None else zero_logit
    )
    hz_gated1 = (
        grad_grad_z[:, :, m0:]
        if grad_grad_z is not None
        else torch.zeros_like(z_gated1)
    )
    h_logit = grad_grad_logit if grad_grad_logit is not None else zero_logit
    if not fold_logit:
        h_logit = h_logit + torch.bmm(hz_scalar, gw)

    weighted = h_logit * d_sig
    d_logit = (
        hz_gated0 * grad_gated0 + fold(hz_gated1 * grad_gated1)
    ) * d_sig + h_logit * grad_sig * dd_sig

    grad_wrt_grad = torch.cat(
        [
            hz_scalar * silu_sig * (1.0 + scalar * (1.0 - silu_sig)),
            hz_gated0 * sig + weighted * z_gated0,
            hz_gated1 * sig.repeat(1, 1, 2) + weighted.repeat(1, 1, 2) * z_gated1,
        ],
        dim=-1,
    )
    grad_wrt_z = torch.cat(
        [
            hz_scalar * grad_scalar * dd_silu + torch.bmm(d_logit, gw.transpose(1, 2)),
            weighted * grad_gated0,
            weighted.repeat(1, 1, 2) * grad_gated1,
        ],
        dim=-1,
    )
    grad_wrt_gw = torch.bmm(scalar.transpose(1, 2), d_logit)
    if not fold_logit:
        grad_wrt_gw = grad_wrt_gw + torch.bmm(
            hz_scalar.transpose(1, 2), grad_sig * d_sig
        )
    return grad_wrt_grad, grad_wrt_z, grad_wrt_gw


if GATED_ACTIVATION_TRITON_AVAILABLE:

    @triton.jit
    def _second_order_kernel(
        hz_ptr,  # (F, E, ROW) cotangent of the pre-activation gradient
        hq_ptr,  # (F, E, L*CF) cotangent of the gate-logit gradient
        g_ptr,  # (F, E, ROW) output cotangent of the layer
        z_ptr,  # (F, E, ROW) pre-activation
        gw_ptr,  # (F, CF, L*CF) gate projection
        gwt_ptr,  # (F, L*CF, CF) transposed gate projection
        dg_ptr,  # (F, E, ROW) gradient w.r.t. the output cotangent
        dz_ptr,  # (F, E, ROW) gradient w.r.t. the pre-activation
        dq_ptr,  # (F, E, L*CF) gradient w.r.t. the gate logit
        hp_ptr,  # (F, E, ROW) running head added onto dg, read when HAS_ADD
        n_edge,
        L: tl.constexpr,
        CF: tl.constexpr,
        FOLD_LOGIT: tl.constexpr,
        HAS_HZ: tl.constexpr,
        HAS_HQ: tl.constexpr,
        HAS_ADD: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Second order of one gated layer, one program per edge block.

        Each gate group is handled in registers: its sigmoid is recomputed from
        the scalar rows, the three value rows sharing it are read once, and both
        the gate and the value contributions are accumulated before any store.
        The two projections against the gate weight are register dots, matching
        the first-order kernel's schedule.
        """
        ROW: tl.constexpr = (3 * L + 1) * CF
        LG: tl.constexpr = L * CF
        CP: tl.constexpr = triton.next_power_of_2(CF)

        pid_m = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        nc = tl.arange(0, CP)
        cm = mm & (nc < CF)[None, :]
        wm = ((nc < CF)[:, None]) & ((nc < CF)[None, :])

        g_row = g_ptr + fid * n_edge * ROW + offs_m * ROW
        z_row = z_ptr + fid * n_edge * ROW + offs_m * ROW
        hz_row = hz_ptr + fid * n_edge * ROW + offs_m * ROW
        dg_row = dg_ptr + fid * n_edge * ROW + offs_m * ROW
        dz_row = dz_ptr + fid * n_edge * ROW + offs_m * ROW
        hp_row = hp_ptr + fid * n_edge * ROW + offs_m * ROW
        hq_row = hq_ptr + (fid * n_edge + offs_m) * LG
        dq_row = dq_ptr + (fid * n_edge + offs_m) * LG

        # === Scalar rows: SiLU value path and the source of every gate ===
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0).to(tl.float32)
        g_s = tl.load(g_row[:, None] + nc[None, :], mask=cm, other=0.0).to(tl.float32)
        if HAS_HZ:
            hz_s = tl.load(hz_row[:, None] + nc[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
        else:
            hz_s = tl.zeros((BLOCK_M, CP), dtype=tl.float32)
        s0 = tl.sigmoid(z_s)
        d_silu = s0 * (1.0 - s0)
        # d/ds of the SiLU derivative.
        dd_silu = d_silu * (2.0 + z_s * (1.0 - 2.0 * s0))
        dg_s = hz_s * s0 * (1.0 + z_s * (1.0 - s0))
        if HAS_ADD:
            dg_s += tl.load(hp_row[:, None] + nc[None, :], mask=cm, other=0.0)
        tl.store(dg_row[:, None] + nc[None, :], dg_s, mask=cm)
        dz_s = hz_s * g_s * dd_silu

        for g in tl.static_range(L):
            gw_g = tl.load(
                gw_ptr + fid * CF * LG + nc[:, None] * LG + (g * CF + nc)[None, :],
                mask=wm,
                other=0.0,
            ).to(tl.float32)
            q_g = tl.dot(z_s, gw_g, input_precision="ieee")
            sig_g = tl.sigmoid(q_g)
            d_sig = sig_g * (1.0 - sig_g)
            dd_sig = d_sig * (1.0 - 2.0 * sig_g)

            # The three value rows that share this gate group.
            r0 = (1 + g) * CF
            rn = ((L + 1) + g) * CF
            rp = ((2 * L + 1) + g) * CF
            g_r0 = tl.load(g_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            g_rn = tl.load(g_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            g_rp = tl.load(g_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            z_r0 = tl.load(z_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            z_rn = tl.load(z_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            z_rp = tl.load(z_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            if HAS_HZ:
                hz_r0 = tl.load(
                    hz_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                hz_rn = tl.load(
                    hz_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                hz_rp = tl.load(
                    hz_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
            else:
                hz_r0 = tl.zeros((BLOCK_M, CP), dtype=tl.float32)
                hz_rn = tl.zeros((BLOCK_M, CP), dtype=tl.float32)
                hz_rp = tl.zeros((BLOCK_M, CP), dtype=tl.float32)

            # Effective logit cotangent: the incoming one plus, when the scalar
            # contraction is folded into this operator, the route through it.
            if HAS_HQ:
                h_q = (
                    tl.load(
                        hq_row[:, None] + (g * CF + nc)[None, :], mask=cm, other=0.0
                    )
                    .to(tl.float32)
                    .to(tl.float32)
                )
            else:
                h_q = tl.zeros((BLOCK_M, CP), dtype=tl.float32)
            if not FOLD_LOGIT:
                h_q = h_q + tl.dot(hz_s, gw_g, input_precision="ieee")
            weighted = h_q * d_sig

            sum_gz = g_r0 * z_r0 + g_rn * z_rn + g_rp * z_rp
            sum_hg = hz_r0 * g_r0 + hz_rn * g_rn + hz_rp * g_rp
            d_q = sum_hg * d_sig + h_q * sum_gz * dd_sig

            dg_r0 = hz_r0 * sig_g + weighted * z_r0
            dg_rn = hz_rn * sig_g + weighted * z_rn
            dg_rp = hz_rp * sig_g + weighted * z_rp
            if HAS_ADD:
                dg_r0 += tl.load(
                    hp_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0
                )
                dg_rn += tl.load(
                    hp_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0
                )
                dg_rp += tl.load(
                    hp_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0
                )
            tl.store(dg_row[:, None] + (r0 + nc)[None, :], dg_r0, mask=cm)
            tl.store(dg_row[:, None] + (rn + nc)[None, :], dg_rn, mask=cm)
            tl.store(dg_row[:, None] + (rp + nc)[None, :], dg_rp, mask=cm)
            tl.store(dz_row[:, None] + (r0 + nc)[None, :], weighted * g_r0, mask=cm)
            tl.store(dz_row[:, None] + (rn + nc)[None, :], weighted * g_rn, mask=cm)
            tl.store(dz_row[:, None] + (rp + nc)[None, :], weighted * g_rp, mask=cm)
            tl.store(dq_row[:, None] + (g * CF + nc)[None, :], d_q, mask=cm)

            gwt_g = tl.load(
                gwt_ptr + fid * LG * CF + (g * CF + nc)[:, None] * CF + nc[None, :],
                mask=wm,
                other=0.0,
            ).to(tl.float32)
            dz_s = tl.dot(d_q, gwt_g, dz_s, input_precision="ieee")

        tl.store(dz_row[:, None] + nc[None, :], dz_s, mask=cm)


def _use_triton(tensor: Tensor) -> bool:
    """Return whether the fused path serves this tensor's device and dtype."""
    return (
        GATED_ACTIVATION_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def gated_activation_second_order(
    grad_grad_z: Tensor | None,
    grad_grad_logit: Tensor | None,
    grad: Tensor,
    z: Tensor,
    gw: Tensor,
    gwt: Tensor,
    lmax: int,
    focus_dim: int,
    fold_logit: bool,
    out_z: Tensor | None = None,
    add_to: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiate one gated layer's backward, fused into a single kernel.

    Parameters
    ----------
    grad_grad_z : Tensor or None
        Cotangent of the pre-activation gradient, with shape ``(F, E, ROW)``.
    grad_grad_logit : Tensor or None
        Cotangent of the gate-logit gradient, with shape ``(F, E, lmax * Cf)``.
    grad : Tensor
        Output cotangent of the layer, with shape ``(F, E, ROW)``.
    z : Tensor
        Pre-activation of the layer, with shape ``(F, E, ROW)``.
    gw : Tensor
        Gate projection, with shape ``(F, Cf, lmax * Cf)``.
    gwt : Tensor
        Transposed gate projection, with shape ``(F, lmax * Cf, Cf)``.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.
    fold_logit : bool
        Whether the caller contracts the gate-logit gradient onto the scalars.
    out_z : Tensor, optional
        Contiguous destination the pre-activation gradient is written into,
        sparing the caller a copy when it lands in a stacked buffer.
    add_to : Tensor, optional
        Running head folded onto the first output in-kernel, sparing the
        caller a separate elementwise addition per layer.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Gradients with respect to ``(grad, z, gw)``.
    """
    if grad_grad_z is None and grad_grad_logit is None:
        zero = torch.zeros_like(grad) if add_to is None else add_to.clone()
        dz = torch.zeros_like(z) if out_z is None else out_z.zero_()
        return zero, dz, torch.zeros_like(gw)
    if not _use_triton(grad):
        result = gated_activation_second_order_reference(
            grad_grad_z,
            grad_grad_logit,
            grad,
            z,
            gw,
            int(lmax),
            int(focus_dim),
            bool(fold_logit),
        )
        head = result[0] if add_to is None else result[0] + add_to
        if out_z is None:
            return head, result[1], result[2]
        out_z.copy_(result[1])
        return head, out_z, result[2]
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    n_focus, n_edge, row = z.shape
    gate_width = lmax * focus_dim
    grad_wrt_grad = torch.empty_like(grad)
    grad_wrt_z = torch.empty_like(z) if out_z is None else out_z
    grad_wrt_logit = torch.empty(
        (n_focus, n_edge, gate_width), device=z.device, dtype=torch.float32
    )
    empty = torch.empty(0, device=z.device, dtype=z.dtype)
    block_m, warps, stages = gated_second_order_config(focus_dim, lmax)
    wrap_triton(_second_order_kernel)[(triton.cdiv(n_edge, block_m), n_focus)](
        grad_grad_z if grad_grad_z is not None else empty,
        grad_grad_logit if grad_grad_logit is not None else empty,
        grad,
        z,
        gw,
        gwt,
        grad_wrt_grad,
        grad_wrt_z,
        grad_wrt_logit,
        add_to if add_to is not None else empty,
        n_edge,
        L=lmax,
        CF=focus_dim,
        FOLD_LOGIT=bool(fold_logit),
        HAS_HZ=grad_grad_z is not None,
        HAS_HQ=grad_grad_logit is not None,
        HAS_ADD=add_to is not None,
        BLOCK_M=block_m,
        num_warps=warps,
        num_stages=stages,
    )
    # The gate weight reduces the whole edge axis, which cuBLAS handles well.
    grad_wrt_gw = torch.bmm(
        z[:, :, :focus_dim].transpose(1, 2), grad_wrt_logit.to(z.dtype)
    )
    if not fold_logit and grad_grad_z is not None:
        # When the scalar contraction is folded in, the scalar cotangent also
        # reaches the gate weight through the first-order gate-logit gradient.
        m0 = (lmax + 1) * focus_dim
        sig = torch.sigmoid(torch.bmm(z[:, :, :focus_dim], gw))
        grad_sig = grad[:, :, focus_dim:m0] * z[:, :, focus_dim:m0] + (
            grad[:, :, m0:] * z[:, :, m0:]
        ).view(n_focus, n_edge, 2, -1).sum(2)
        grad_wrt_gw = grad_wrt_gw + torch.bmm(
            grad_grad_z[:, :, :focus_dim].transpose(1, 2),
            grad_sig * sig * (1.0 - sig),
        )
    return grad_wrt_grad, grad_wrt_z, grad_wrt_gw
