# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Attention utilities for DPA4/SeZM message passing.

This module is the dpmodel port of
``deepmd.pt.model.descriptor.sezm_nn.attention``. It implements the
destination-wise envelope-gated softmax used by the SO(2) attention path.

Padded-edge adaptation
----------------------
The pt version consumes a sparse edge list and reduces per destination node
with ``scatter_reduce(amax)`` / ``scatter_add`` keyed by ``dst``. In the
dpmodel padded layout (see ``edge_cache.EdgeCache``) the edge axis is
``E = n_nodes * nnei`` with slot ``(i, j)`` belonging to node ``i``, so every
destination-wise reduction becomes a plain reduction over the ``nnei`` axis
after a ``(n_nodes, nnei, ...)`` reshape, and invalid (padded) slots are
removed by folding ``edge_mask`` into the non-negative per-edge weight.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.utils.network import (
    softplus_t,
)


def segment_envelope_gated_softmax(
    logits: Any,
    edge_env: Any,
    n_nodes: int,
    z_bias_raw: Any,
    eps: float,
    src_weight: Any = None,
    edge_mask: Any = None,
) -> Any:
    """
    Compute destination-wise envelope-gated softmax attention.

    All array arguments must live in the same array namespace.

    Parameters
    ----------
    logits
        Attention logits with shape (E, F, H), padded-edge layout with
        ``E = n_nodes * nnei``.
    edge_env
        Cutoff envelope weights with shape (E, 1) or (E,).
    n_nodes
        Number of nodes. The pt ``dst`` argument is dropped: in the padded
        layout the destination of edge slot ``(i, j)`` is implicitly node
        ``i``.
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
        weight, so invalid slots drop out of the group max, the numerator,
        and the denominator exactly like absent edges in the pt sparse
        layout.

    Returns
    -------
    Array
        Normalized edge weights with shape (E, F, H). Zero on invalid slots.
    """
    xp = array_api_compat.array_namespace(logits)
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    eps_f = float(eps)
    if n_nodes <= 0 or n_edge % int(n_nodes) != 0:
        raise ValueError(
            "padded-edge layout requires E to be a multiple of n_nodes; "
            f"got E={n_edge}, n_nodes={n_nodes}"
        )
    nnei = n_edge // int(n_nodes)
    device = array_api_compat.device(logits)

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
    # pt: scatter_reduce(amax) over dst — padded-edge max over the nnei axis.
    group_max = xp.max(
        xp.reshape(logits_for_max, (n_nodes, nnei, n_channel)), axis=1
    )  # (N, n_channel)
    edge_max = xp.reshape(
        xp.broadcast_to(group_max[:, None, :], (n_nodes, nnei, n_channel)),
        (n_edge, n_channel),
    )
    zeros_en = xp.zeros((n_edge, n_channel), dtype=logits.dtype, device=device)
    zeros_nn = xp.zeros((n_nodes, n_channel), dtype=logits.dtype, device=device)
    edge_max = xp.where(xp.isfinite(edge_max), edge_max, zeros_en)
    group_max_safe = xp.where(xp.isfinite(group_max), group_max, zeros_nn)

    # === Step 3. Envelope/SFPG-gated exponential terms ===
    exp_shifted = xp.exp(logits_2d - edge_max)
    edge_weighted_exp = edge_weight_sq[:, None] * exp_shifted

    # === Step 4. Destination-wise normalization with positive denominator bias ===
    # pt: scatter_add over dst — padded-edge masked sum over the nnei axis
    # (invalid slots already carry zero weight).
    denom_sum = xp.sum(
        xp.reshape(edge_weighted_exp, (n_nodes, nnei, n_channel)), axis=1
    )  # (N, n_channel)
    denom = denom_sum + zeta * xp.exp(-group_max_safe)

    denom_edge = xp.reshape(
        xp.broadcast_to(denom[:, None, :], (n_nodes, nnei, n_channel)),
        (n_edge, n_channel),
    )
    alpha = edge_weighted_exp / (denom_edge + eps_f)
    return xp.reshape(alpha, (n_edge, n_focus, n_head))
