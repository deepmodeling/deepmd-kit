# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Attention utilities for DPA4/SeZM message passing.

This module implements the destination-wise envelope-gated softmax used by the
SO(2) attention path in the SeZM descriptor.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.attention``.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    xp_add_at,
    xp_maximum_at,
)
from deepmd.dpmodel.utils.network import (
    softplus_t,
)


def segment_envelope_gated_softmax(
    logits: Any,
    edge_env: Any,
    dst: Any,
    n_nodes: int,
    z_bias_raw: Any,
    eps: float,
    src_weight: Any = None,
    edge_mask: Any = None,
) -> Any:
    """
    Compute destination-wise envelope-gated softmax attention.

    Parameters
    ----------
    logits
        Attention logits with shape (E, F, H).
    edge_env
        Cutoff envelope weights with shape (E, 1) or (E,).
    dst
        Destination node indices with shape (E,). The group max and the
        denominator sum are scattered over these indices, which makes the
        normalization layout-agnostic: it is correct both for the padded
        ``call`` (where ``dst == repeat(arange(n_nodes), nnei)``) and for the
        sparse ``call_with_edges`` (arbitrary ``dst`` order and per-node
        degree).
    n_nodes
        Number of nodes.
    z_bias_raw
        Unconstrained denominator bias with shape (F, H).
        Softplus is applied to keep the bias strictly positive.
    eps
        Small epsilon for denominator stability.
    src_weight
        Optional per-edge source-side multiplier with shape (E, 1) or
        (E,). When provided the per-edge weight becomes
        ``edge_env**2 * src_weight`` and the attention reduces to
        ``edge_env**2 * src_weight * exp(logits) /
        (zeta + sum(edge_env**2 * src_weight * exp(logits)))``.
        ``src_weight = 0`` therefore removes the source from both the
        numerator and the denominator, which is what SFPG needs so that
        a muted source does not even leak through the softmax
        normalization.
    edge_mask
        Optional padded-edge validity mask with shape (E,) or (E, 1);
        zero marks invalid slots. Folded into the non-negative per-edge
        weight so invalid slots drop out of the group max, the numerator,
        and the denominator.

    Returns
    -------
    Array
        Normalized edge weights with shape (E, F, H). Zero on invalid slots.
    """
    xp = array_api_compat.array_namespace(logits)
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    eps_f = float(eps)
    device = array_api_compat.device(logits)
    dst = xp.astype(dst, xp.int64)

    # === Step 1. Flatten (F, H) and build the effective per-edge weight ===
    logits_2d = xp.reshape(logits, (n_edge, n_channel))
    zeros_e = xp.zeros((n_edge,), dtype=logits.dtype, device=device)
    edge_env_1d = xp.astype(xp.reshape(edge_env, (n_edge,)), logits.dtype)
    edge_env_1d = xp.where(edge_env_1d > 0.0, edge_env_1d, zeros_e)
    # edge_weight_sq acts as the non-negative multiplier applied to every
    # ``exp(logit)`` term. Folding ``src_weight`` (and, in the padded
    # layout, ``edge_mask``) here guarantees that any edge with zero weight
    # is excluded from the group max, the numerator, and the denominator in
    # a single pass.
    edge_weight_sq = edge_env_1d * edge_env_1d
    if src_weight is not None:
        src_weight_1d = xp.astype(xp.reshape(src_weight, (n_edge,)), logits.dtype)
        src_weight_1d = xp.where(src_weight_1d > 0.0, src_weight_1d, zeros_e)
        edge_weight_sq = edge_weight_sq * src_weight_1d
    if edge_mask is not None:
        mask_1d = xp.astype(xp.reshape(edge_mask, (n_edge,)), logits.dtype)
        edge_weight_sq = edge_weight_sq * mask_1d
    zeta = xp.astype(xp.reshape(softplus_t(z_bias_raw), (1, n_channel)), logits.dtype)
    has_weight = edge_weight_sq > 0.0
    minus_inf = xp.full(
        (n_edge, n_channel),
        float("-inf"),
        dtype=logits.dtype,
        device=device,
    )
    logits_for_max = xp.where(
        has_weight[:, None],
        logits_2d,
        minus_inf,
    )

    # === Step 2. Destination-wise max for stable exponentials ===
    # Destination segment max over ``dst`` (pt ``scatter_reduce`` amax). The
    # scatter is layout-agnostic and the maximum is order-independent, so the
    # padded ``call`` stays bit-exact while the sparse ``call_with_edges`` is
    # handled by the same code path.
    group_max = xp_maximum_at(
        xp.full((n_nodes, n_channel), float("-inf"), dtype=logits.dtype, device=device),
        dst,
        logits_for_max,
    )  # (N, n_channel)
    edge_max = xp.take(group_max, dst, axis=0)
    zeros_en = xp.zeros((n_edge, n_channel), dtype=logits.dtype, device=device)
    zeros_nn = xp.zeros((n_nodes, n_channel), dtype=logits.dtype, device=device)
    edge_max = xp.where(xp.isfinite(edge_max), edge_max, zeros_en)
    group_max_safe = xp.where(xp.isfinite(group_max), group_max, zeros_nn)

    # === Step 3. Envelope/SFPG-gated exponential terms ===
    exp_shifted = xp.exp(logits_2d - edge_max)
    edge_weighted_exp = edge_weight_sq[:, None] * exp_shifted

    # === Step 4. Destination-wise normalization with positive denominator bias ===
    # Destination segment sum over ``dst`` (pt ``scatter_add``); invalid slots
    # already carry zero weight. Layout-agnostic like the group max above.
    denom_sum = xp_add_at(
        xp.zeros((n_nodes, n_channel), dtype=logits.dtype, device=device),
        dst,
        edge_weighted_exp,
    )  # (N, n_channel)
    denom = denom_sum + zeta * xp.exp(-group_max_safe)

    denom_edge = xp.take(denom, dst, axis=0)
    alpha = edge_weighted_exp / (denom_edge + eps_f)
    return xp.reshape(alpha, (n_edge, n_focus, n_head))
