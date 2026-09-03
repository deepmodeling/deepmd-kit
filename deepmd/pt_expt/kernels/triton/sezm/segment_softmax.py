# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
r"""Fused destination-segmented softmax for the SeZM attention weights.

The attention path normalizes per-edge logits over each destination node's
incoming edges together with a per-channel null mass (see
``segment_envelope_gated_softmax``).  Expressed in ATen the forward is a
``scatter_reduce`` / ``scatter_add`` / ``index_select`` chain whose backward
and second order the force-loss trace expands into a dozen materialized
surfaces and serialized scatters per convolution block.  This module runs the
whole normalization as one CSR-segmented operator per direction: one kernel
for the forward, one for the first-order backward, one for the second order,
each walking the destination-sorted edge list the flash aggregation already
maintains.

Mathematics
-----------
Per destination segment and channel, with :math:`l_e` the effective logit
(:math:`\text{logit}_e + 2\ln \text{env}_e`; edges with a non-positive
envelope are excluded), :math:`l_0` the null logit, and
:math:`w = \exp(l - m)` in the shared shifted frame,

.. math::

    \alpha_e = w_e / D, \qquad D = w_0 + \textstyle\sum_e w_e .

First order, given the output cotangent :math:`\bar g_e` and
:math:`S = \sum_e \alpha_e \bar g_e`:

.. math::

    \partial l_e = \alpha_e (\bar g_e - S), \qquad
    \partial l_0 = -\alpha_0 S, \qquad
    \partial \text{env}_e = \partial l_e \cdot 2 / \text{env}_e .

Second order, given cotangents :math:`h_e` of :math:`\partial l_e`,
:math:`h^{env}_e` of :math:`\partial \text{env}_e` and :math:`h_0` of
:math:`\partial l_0`.  The envelope cotangent first folds onto the logit
cotangent, :math:`h_e \mathrel{+}= h^{env}_e \cdot 2/\text{env}_e`, leaving a
direct curvature term :math:`-h^{env}_e \, \partial l_e \cdot
2/\text{env}_e^2` on the envelope.  With the segment scalars

.. math::

    T = \textstyle\sum_e h_e \alpha_e, \quad
    U = \textstyle\sum_e h_e \alpha_e \bar g_e, \quad
    Q = U - 2TS - 2 h_0 \alpha_0 S,

the gradients are

.. math::

    \partial \bar g_e &= \alpha_e (h_e - T - h_0 \alpha_0), \\
    q_e &= h_e \bar g_e - h_e S - (T + h_0 \alpha_0) \bar g_e, \\
    \partial l_e &= \alpha_e (q_e - Q), \qquad
    \partial l_0 = \alpha_0 (-h_0 S - Q).

Being the highest order required, nothing differentiates the second-order
body in turn.

Layout contract
---------------
Channels are the trailing axis with width ``C = F * H`` (a handful in
production); each program owns one destination segment and holds the channel
vector in registers.  ``alpha`` and the per-node ``alpha_null`` surface are
saved by the forward, so neither backward re-reduces the segment maxima.
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

__all__ = [
    "SEGMENT_SOFTMAX_TRITON_AVAILABLE",
    "segment_softmax",
]

try:
    import triton
    import triton.language as tl

    SEGMENT_SOFTMAX_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    SEGMENT_SOFTMAX_TRITON_AVAILABLE = False

_MAX_CHANNELS = 16
_SEG_BLOCK = 32  # edges reduced per inner tile; segments iterate in tiles


def _segment_softmax_reference(
    logits: Tensor,
    edge_env: Tensor,
    null_logit: Tensor,
    dst: Tensor,
    n_nodes: int,
) -> Tensor:
    """Eager ground truth: the scatter/gather softmax with a null mass.

    Parameters
    ----------
    logits : Tensor
        Attention logits with shape ``(E, C)``, float32.
    edge_env : Tensor
        Cutoff envelope with shape ``(E,)``, float32; non-positive entries
        exclude the edge.
    null_logit : Tensor
        Per-channel null logit with shape ``(C,)``, float32.
    dst : Tensor
        Destination node index with shape ``(E,)``.
    n_nodes : int
        Number of destination nodes.

    Returns
    -------
    Tensor
        Normalized weights with shape ``(E, C)``, float32.
    """
    n_edge, n_channel = logits.shape
    positive = edge_env > 0.0
    safe_env = torch.where(positive, edge_env, torch.ones_like(edge_env))
    eff = torch.where(
        positive.unsqueeze(1),
        logits + 2.0 * torch.log(safe_env).unsqueeze(1),
        torch.full_like(logits, float("-inf")),
    )
    dst_index = dst.reshape(n_edge, 1).expand(n_edge, n_channel)
    group_max = null_logit.expand(n_nodes, n_channel).clone()
    group_max = torch.scatter_reduce(
        group_max, 0, dst_index, eff, reduce="amax", include_self=True
    )
    edge_exp = torch.exp(eff - group_max.index_select(0, dst))
    denom = torch.zeros(n_nodes, n_channel, dtype=logits.dtype, device=logits.device)
    denom = torch.scatter_add(denom, 0, dst_index, edge_exp)
    denom = denom + torch.exp(null_logit.unsqueeze(0) - group_max)
    return edge_exp / denom.index_select(0, dst)


if SEGMENT_SOFTMAX_TRITON_AVAILABLE:

    @triton.jit
    def _seg_softmax_fwd_kernel(
        logits_ptr,  # (E, C) attention logits
        env_ptr,  # (E,) cutoff envelope
        null_ptr,  # (C,) null logit
        order_ptr,  # (E,) destination-sorted edge order
        rowptr_ptr,  # (N + 1,) CSR row pointers
        alpha_ptr,  # (E, C) output weights
        anull_ptr,  # (N, C) output null weights
        C: tl.constexpr,
        CP: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """One program per destination segment: max, denominator, weights."""
        node = tl.program_id(0).to(tl.int64)
        beg = tl.load(rowptr_ptr + node).to(tl.int64)
        end = tl.load(rowptr_ptr + node + 1).to(tl.int64)

        nc = tl.arange(0, CP)
        c_mask = nc < C
        null = tl.load(null_ptr + nc, mask=c_mask, other=0.0)

        # === Pass 1. Segment maximum in the shared shifted frame ===
        m = null
        offs = tl.arange(0, BLOCK_E)
        for tile in range(beg, end, BLOCK_E):
            idx = tile + offs
            e_mask = idx < end
            edge = tl.load(order_ptr + idx, mask=e_mask, other=0).to(tl.int64)
            env = tl.load(env_ptr + edge, mask=e_mask, other=0.0)
            active = e_mask & (env > 0.0)
            lw = 2.0 * tl.log(tl.where(active, env, 1.0))
            lg = tl.load(
                logits_ptr + edge[:, None] * C + nc[None, :],
                mask=active[:, None] & c_mask[None, :],
                other=float("-inf"),
            )
            eff = tl.where(active[:, None], lg + lw[:, None], float("-inf"))
            m = tl.maximum(m, tl.max(eff, axis=0))

        # === Pass 2. Denominator including the null mass ===
        denom = tl.exp(null - m)
        for tile in range(beg, end, BLOCK_E):
            idx = tile + offs
            e_mask = idx < end
            edge = tl.load(order_ptr + idx, mask=e_mask, other=0).to(tl.int64)
            env = tl.load(env_ptr + edge, mask=e_mask, other=0.0)
            active = e_mask & (env > 0.0)
            lw = 2.0 * tl.log(tl.where(active, env, 1.0))
            lg = tl.load(
                logits_ptr + edge[:, None] * C + nc[None, :],
                mask=active[:, None] & c_mask[None, :],
                other=float("-inf"),
            )
            eff = tl.where(active[:, None], lg + lw[:, None], float("-inf"))
            denom += tl.sum(tl.exp(eff - m[None, :]), axis=0)

        tl.store(anull_ptr + node * C + nc, tl.exp(null - m) / denom, mask=c_mask)

        # === Pass 3. Normalized weights ===
        for tile in range(beg, end, BLOCK_E):
            idx = tile + offs
            e_mask = idx < end
            edge = tl.load(order_ptr + idx, mask=e_mask, other=0).to(tl.int64)
            env = tl.load(env_ptr + edge, mask=e_mask, other=0.0)
            active = e_mask & (env > 0.0)
            lw = 2.0 * tl.log(tl.where(active, env, 1.0))
            lg = tl.load(
                logits_ptr + edge[:, None] * C + nc[None, :],
                mask=active[:, None] & c_mask[None, :],
                other=float("-inf"),
            )
            eff = tl.where(active[:, None], lg + lw[:, None], float("-inf"))
            alpha = tl.exp(eff - m[None, :]) / denom[None, :]
            tl.store(
                alpha_ptr + edge[:, None] * C + nc[None, :],
                alpha,
                mask=e_mask[:, None] & c_mask[None, :],
            )

    @triton.jit
    def _seg_softmax_bwd_kernel(
        galpha_ptr,  # (E, C) output cotangent
        alpha_ptr,  # (E, C) saved weights
        anull_ptr,  # (N, C) saved null weights
        env_ptr,  # (E,) cutoff envelope
        order_ptr,
        rowptr_ptr,
        glogit_ptr,  # (E, C) logit gradient
        genv_ptr,  # (E,) envelope gradient
        gnull_ptr,  # (N, C) per-node null-logit gradient
        C: tl.constexpr,
        CP: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """First order: ``glogit = alpha * (galpha - S)`` per segment."""
        node = tl.program_id(0).to(tl.int64)
        beg = tl.load(rowptr_ptr + node).to(tl.int64)
        end = tl.load(rowptr_ptr + node + 1).to(tl.int64)

        nc = tl.arange(0, CP)
        c_mask = nc < C
        offs = tl.arange(0, BLOCK_E)

        # === Pass 1. S = sum(alpha * galpha) ===
        s_vec = tl.zeros((CP,), dtype=tl.float32)
        for tile in range(beg, end, BLOCK_E):
            idx = tile + offs
            e_mask = idx < end
            edge = tl.load(order_ptr + idx, mask=e_mask, other=0).to(tl.int64)
            em = e_mask[:, None] & c_mask[None, :]
            alpha = tl.load(
                alpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            galpha = tl.load(
                galpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            s_vec += tl.sum(alpha * galpha, axis=0)

        anull = tl.load(anull_ptr + node * C + nc, mask=c_mask, other=0.0)
        tl.store(gnull_ptr + node * C + nc, -anull * s_vec, mask=c_mask)

        # === Pass 2. Per-edge gradients ===
        for tile in range(beg, end, BLOCK_E):
            idx = tile + offs
            e_mask = idx < end
            edge = tl.load(order_ptr + idx, mask=e_mask, other=0).to(tl.int64)
            em = e_mask[:, None] & c_mask[None, :]
            alpha = tl.load(
                alpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            galpha = tl.load(
                galpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            glogit = alpha * (galpha - s_vec[None, :])
            tl.store(glogit_ptr + edge[:, None] * C + nc[None, :], glogit, mask=em)
            env = tl.load(env_ptr + edge, mask=e_mask, other=1.0)
            env_safe = tl.where(env > 0.0, env, 1.0)
            genv = tl.sum(glogit, axis=1) * 2.0 / env_safe
            tl.store(
                genv_ptr + edge,
                tl.where(env > 0.0, genv, 0.0),
                mask=e_mask,
            )

    @triton.jit
    def _seg_softmax_2nd_kernel(
        h_ptr,  # (E, C) cotangent of the logit gradient
        henv_ptr,  # (E,) cotangent of the envelope gradient
        hnull_ptr,  # (N, C) cotangent of the null-logit gradient
        galpha_ptr,  # (E, C) output cotangent of the layer
        alpha_ptr,  # (E, C) saved weights
        anull_ptr,  # (N, C) saved null weights
        env_ptr,  # (E,) cutoff envelope
        order_ptr,
        rowptr_ptr,
        dgalpha_ptr,  # (E, C) gradient w.r.t. galpha
        dlogit_ptr,  # (E, C) gradient w.r.t. the logits
        denv_ptr,  # (E,) gradient w.r.t. the envelope
        dnull_ptr,  # (N, C) per-node gradient w.r.t. the null logit
        C: tl.constexpr,
        CP: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Second order of the segmented softmax backward (see module docs)."""
        node = tl.program_id(0).to(tl.int64)
        beg = tl.load(rowptr_ptr + node).to(tl.int64)
        end = tl.load(rowptr_ptr + node + 1).to(tl.int64)

        nc = tl.arange(0, CP)
        c_mask = nc < C
        offs = tl.arange(0, BLOCK_E)

        # === Pass 1. Segment scalars S, T, U ===
        s_vec = tl.zeros((CP,), dtype=tl.float32)
        t_vec = tl.zeros((CP,), dtype=tl.float32)
        u_vec = tl.zeros((CP,), dtype=tl.float32)
        for tile in range(beg, end, BLOCK_E):
            idx = tile + offs
            e_mask = idx < end
            edge = tl.load(order_ptr + idx, mask=e_mask, other=0).to(tl.int64)
            em = e_mask[:, None] & c_mask[None, :]
            alpha = tl.load(
                alpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            galpha = tl.load(
                galpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            h = tl.load(h_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0)
            env = tl.load(env_ptr + edge, mask=e_mask, other=1.0)
            env_safe = tl.where(env > 0.0, env, 1.0)
            henv = tl.load(henv_ptr + edge, mask=e_mask, other=0.0)
            # The envelope cotangent folds onto the logit cotangent, uniformly
            # over channels: g_env = sum_c(glogit) * 2 / env.
            h = h + (tl.where(env > 0.0, henv * 2.0 / env_safe, 0.0))[:, None]
            s_vec += tl.sum(alpha * galpha, axis=0)
            t_vec += tl.sum(alpha * h, axis=0)
            u_vec += tl.sum(alpha * h * galpha, axis=0)

        anull = tl.load(anull_ptr + node * C + nc, mask=c_mask, other=0.0)
        h0 = tl.load(hnull_ptr + node * C + nc, mask=c_mask, other=0.0)
        h0a0 = h0 * anull
        q_seg = u_vec - 2.0 * t_vec * s_vec - 2.0 * h0a0 * s_vec
        tl.store(
            dnull_ptr + node * C + nc,
            anull * (-h0 * s_vec - q_seg),
            mask=c_mask,
        )

        # === Pass 2. Per-edge outputs ===
        for tile in range(beg, end, BLOCK_E):
            idx = tile + offs
            e_mask = idx < end
            edge = tl.load(order_ptr + idx, mask=e_mask, other=0).to(tl.int64)
            em = e_mask[:, None] & c_mask[None, :]
            alpha = tl.load(
                alpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            galpha = tl.load(
                galpha_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0
            )
            h_raw = tl.load(h_ptr + edge[:, None] * C + nc[None, :], mask=em, other=0.0)
            env = tl.load(env_ptr + edge, mask=e_mask, other=1.0)
            env_safe = tl.where(env > 0.0, env, 1.0)
            active = env > 0.0
            henv = tl.load(henv_ptr + edge, mask=e_mask, other=0.0)
            fold = tl.where(active, henv * 2.0 / env_safe, 0.0)
            h = h_raw + fold[:, None]

            dgalpha = alpha * (h - t_vec[None, :] - h0a0[None, :])
            tl.store(dgalpha_ptr + edge[:, None] * C + nc[None, :], dgalpha, mask=em)

            q = h * galpha - h * s_vec[None, :] - (t_vec + h0a0)[None, :] * galpha
            dlogit = alpha * (q - q_seg[None, :])
            tl.store(dlogit_ptr + edge[:, None] * C + nc[None, :], dlogit, mask=em)

            # Envelope gradient: the chain through the folded logit cotangent
            # plus the direct curvature of ``2 / env``.
            glogit = alpha * (galpha - s_vec[None, :])
            denv = tl.sum(dlogit, axis=1) * 2.0 / env_safe - henv * tl.sum(
                glogit, axis=1
            ) * 2.0 / (env_safe * env_safe)
            tl.store(denv_ptr + edge, tl.where(active, denv, 0.0), mask=e_mask)


def _use_triton(tensor: Tensor) -> bool:
    """Return whether the fused path serves this tensor's device and dtype."""
    return (
        SEGMENT_SOFTMAX_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype is torch.float32
    )


def _seg_softmax_impl(
    logits: Tensor,
    edge_env: Tensor,
    null_logit: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
) -> tuple[Tensor, Tensor]:
    """Forward: normalized weights and the per-node null weight."""
    n_edge, n_channel = logits.shape
    n_nodes = row_ptr.shape[0] - 1
    if not _use_triton(logits):
        alpha = _segment_softmax_reference(logits, edge_env, null_logit, dst, n_nodes)
        # The null weight is recomputed cheaply on the reference path.
        positive = edge_env > 0.0
        safe_env = torch.where(positive, edge_env, torch.ones_like(edge_env))
        eff = torch.where(
            positive.unsqueeze(1),
            logits + 2.0 * torch.log(safe_env).unsqueeze(1),
            torch.full_like(logits, float("-inf")),
        )
        dst_index = dst.reshape(n_edge, 1).expand(n_edge, n_channel)
        group_max = null_logit.expand(n_nodes, n_channel).clone()
        group_max = torch.scatter_reduce(
            group_max, 0, dst_index, eff, reduce="amax", include_self=True
        )
        edge_exp = torch.exp(eff - group_max.index_select(0, dst))
        denom = torch.zeros(
            n_nodes, n_channel, dtype=logits.dtype, device=logits.device
        )
        denom = torch.scatter_add(denom, 0, dst_index, edge_exp)
        null_mass = torch.exp(null_logit.unsqueeze(0) - group_max)
        return alpha, null_mass / (denom + null_mass)
    alpha = torch.empty_like(logits)
    alpha_null = torch.empty(
        (n_nodes, n_channel), device=logits.device, dtype=logits.dtype
    )
    if n_edge == 0:
        alpha_null.copy_(torch.ones_like(alpha_null))
        return alpha, alpha_null
    cp = triton.next_power_of_2(max(n_channel, 2))
    wrap_triton(_seg_softmax_fwd_kernel)[(n_nodes,)](
        logits,
        edge_env,
        null_logit,
        order,
        row_ptr,
        alpha,
        alpha_null,
        C=n_channel,
        CP=cp,
        BLOCK_E=_SEG_BLOCK,
        num_warps=1,
        num_stages=2,
    )
    return alpha, alpha_null


def _seg_softmax_bwd_impl(
    galpha: Tensor,
    logits: Tensor,
    edge_env: Tensor,
    null_logit: Tensor,
    alpha: Tensor,
    alpha_null: Tensor,
    order: Tensor,
    row_ptr: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """First-order backward: logit, envelope and per-node null gradients.

    ``logits`` and ``null_logit`` are carried as explicit operands although
    the kernel reads only the saved weights: the second-order formula returns
    the *total* logit and null-logit gradients (the analytic expressions
    already traverse the weights' dependence on them), so those inputs give
    the cotangents their autograd edge while the weight operands are declared
    non-differentiable.
    """
    n_edge, n_channel = galpha.shape
    n_nodes = row_ptr.shape[0] - 1
    glogit = torch.empty_like(galpha)
    genv = torch.empty_like(edge_env)
    gnull = torch.empty((n_nodes, n_channel), device=galpha.device, dtype=galpha.dtype)
    if n_edge == 0:
        gnull.zero_()
        return glogit, genv, gnull
    cp = triton.next_power_of_2(max(n_channel, 2))
    wrap_triton(_seg_softmax_bwd_kernel)[(n_nodes,)](
        galpha.contiguous(),
        alpha,
        alpha_null,
        edge_env,
        order,
        row_ptr,
        glogit,
        genv,
        gnull,
        C=n_channel,
        CP=cp,
        BLOCK_E=_SEG_BLOCK,
        num_warps=1,
        num_stages=2,
    )
    return glogit, genv, gnull


def _seg_softmax_2nd_impl(
    h_logit: Tensor,
    h_env: Tensor,
    h_null: Tensor,
    galpha: Tensor,
    alpha: Tensor,
    alpha_null: Tensor,
    edge_env: Tensor,
    order: Tensor,
    row_ptr: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Second order of the backward (see the module docstring).

    The incoming cotangents may be broadcast views (the null cotangent in
    particular arrives expanded from the reduced ``(C,)`` gradient), so every
    kernel operand is compacted to the flat layout the pointers assume.
    """
    n_edge, n_channel = galpha.shape
    n_nodes = row_ptr.shape[0] - 1
    dgalpha = torch.empty_like(galpha)
    dlogit = torch.empty_like(galpha)
    denv = torch.empty_like(edge_env)
    dnull = torch.empty((n_nodes, n_channel), device=galpha.device, dtype=galpha.dtype)
    if n_edge == 0:
        dnull.zero_()
        return dgalpha, dlogit, denv, dnull
    cp = triton.next_power_of_2(max(n_channel, 2))
    wrap_triton(_seg_softmax_2nd_kernel)[(n_nodes,)](
        h_logit.contiguous(),
        h_env.contiguous(),
        h_null.contiguous(),
        galpha.contiguous(),
        alpha,
        alpha_null,
        edge_env,
        order,
        row_ptr,
        dgalpha,
        dlogit,
        denv,
        dnull,
        C=n_channel,
        CP=cp,
        BLOCK_E=_SEG_BLOCK,
        num_warps=1,
        num_stages=2,
    )
    return dgalpha, dlogit, denv, dnull


_seg_softmax_op = torch.library.triton_op(
    "sezm_triton::segment_softmax", mutates_args=()
)(_seg_softmax_impl)
_seg_softmax_bwd_op = torch.library.triton_op(
    "sezm_triton::segment_softmax_bwd", mutates_args=()
)(_seg_softmax_bwd_impl)
_seg_softmax_2nd_op = torch.library.triton_op(
    "sezm_triton::segment_softmax_2nd", mutates_args=()
)(_seg_softmax_2nd_impl)


@_seg_softmax_op.register_fake
def _(logits, edge_env, null_logit, order, row_ptr, dst):
    n_nodes = row_ptr.shape[0] - 1
    return (
        torch.empty_like(logits),
        logits.new_empty((n_nodes, logits.shape[1])),
    )


@_seg_softmax_bwd_op.register_fake
def _(galpha, logits, edge_env, null_logit, alpha, alpha_null, order, row_ptr):
    return (
        torch.empty_like(galpha),
        torch.empty_like(edge_env),
        torch.empty_like(alpha_null),
    )


@_seg_softmax_2nd_op.register_fake
def _(h_logit, h_env, h_null, galpha, alpha, alpha_null, edge_env, order, row_ptr):
    return (
        torch.empty_like(galpha),
        torch.empty_like(galpha),
        torch.empty_like(edge_env),
        torch.empty_like(alpha_null),
    )


def _seg_softmax_setup_context(ctx, inputs, output):
    logits, edge_env, null_logit, order, row_ptr, dst = inputs
    alpha, alpha_null = output
    ctx.save_for_backward(
        logits, edge_env, null_logit, alpha, alpha_null, order, row_ptr
    )


def _seg_softmax_backward(ctx, galpha, galpha_null):
    """First order; the saved weights make the segment maxima unnecessary.

    The null-weight output exists to carry state to this backward and is not
    a consumer-facing quantity; its cotangent is structurally zero (a tracer
    materializes it as zeros), so it does not enter the formula.
    """
    logits, edge_env, null_logit, alpha, alpha_null, order, row_ptr = ctx.saved_tensors
    glogit, genv, gnull = _seg_softmax_bwd_op(
        galpha, logits, edge_env, null_logit, alpha, alpha_null, order, row_ptr
    )
    # The per-node null gradients reduce to the (C,) null-logit gradient in
    # ATen, which stays differentiable for the second order.
    return glogit, genv, gnull.sum(dim=0), None, None, None


_seg_softmax_op.register_autograd(
    _seg_softmax_backward, setup_context=_seg_softmax_setup_context
)


def _seg_softmax_bwd_setup_context(ctx, inputs, output):
    galpha, logits, edge_env, null_logit, alpha, alpha_null, order, row_ptr = inputs
    ctx.save_for_backward(galpha, alpha, alpha_null, edge_env, order, row_ptr)


def _seg_softmax_bwd_backward(ctx, h_logit, h_env, h_null):
    """Second order of the segmented softmax.

    The analytic expressions return the *total* gradients with respect to the
    logits, the envelope and the null logit -- their traversal of the saved
    weights' dependence on those inputs is already folded in -- so the weight
    operands receive no cotangent of their own.
    """
    galpha, alpha, alpha_null, edge_env, order, row_ptr = ctx.saved_tensors
    dgalpha, dlogit, denv, dnull = _seg_softmax_2nd_op(
        h_logit,
        h_env,
        h_null,
        galpha,
        alpha,
        alpha_null,
        edge_env,
        order,
        row_ptr,
    )
    # inputs: galpha, logits, edge_env, null_logit, alpha, alpha_null,
    # order, row_ptr.
    return dgalpha, dlogit, denv, dnull.sum(dim=0), None, None, None, None


_seg_softmax_bwd_op.register_autograd(
    _seg_softmax_bwd_backward, setup_context=_seg_softmax_bwd_setup_context
)


def segment_softmax(
    logits: Tensor,
    edge_env: Tensor,
    null_logit: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    dst: Tensor,
) -> Tensor:
    """Destination-segmented softmax with a per-channel null mass.

    Parameters
    ----------
    logits : Tensor
        Attention logits with shape ``(E, C)``, float32.
    edge_env : Tensor
        Cutoff envelope with shape ``(E,)``, float32.  The per-edge mass is
        ``env**2 * exp(logits)``; non-positive entries exclude the edge.
    null_logit : Tensor
        Per-channel null logit with shape ``(C,)``, float32.
    order : Tensor
        Destination-sorted edge order with shape ``(E,)``.
    row_ptr : Tensor
        CSR row pointers with shape ``(N + 1,)``.
    dst : Tensor
        Destination node index with shape ``(E,)``.

    Returns
    -------
    Tensor
        Normalized weights with shape ``(E, C)``, float32.
    """
    alpha, _ = _seg_softmax_op(logits, edge_env, null_logit, order, row_ptr, dst)
    return alpha
