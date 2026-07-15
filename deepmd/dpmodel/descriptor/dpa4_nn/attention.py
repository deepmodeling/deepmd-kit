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
        Small positive floor added to the physical null mass.
    src_weight
        Optional per-edge source-side multiplier with shape (E, 1) or
        (E,). When provided, the physical per-edge mass is
        ``edge_env**2 * src_weight * exp(logits)`` and the denominator is the
        sum of edge masses plus the positive null mass
        ``softplus(z_bias_raw) + eps``.
        ``src_weight = 0`` therefore removes the source from both the
        numerator and the denominator, which is what SFPG needs so that
        a muted source does not even leak through the softmax
        normalization.
    edge_mask
        Optional binary padded-edge validity mask with shape (E,) or (E, 1);
        one marks valid slots and zero marks invalid slots. Invalid slots drop
        out of the group max, the numerator, and the denominator.

    Returns
    -------
    Array
        Normalized edge weights with shape (E, F, H). Zero on invalid slots.
    """
    xp = array_api_compat.array_namespace(logits)
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    device = array_api_compat.device(logits)
    input_dtype = logits.dtype
    promote = "float16" in str(input_dtype)
    compute_dtype = xp.float32 if promote else input_dtype
    dst = xp.astype(dst, xp.int64)

    # === Step 1. Build factor-wise effective logits ===
    # Computing the logarithms before multiplying the factors avoids losing a
    # physically nonzero edge when ``edge_env**2 * src_weight * edge_mask``
    # underflows.
    logits_2d = xp.astype(xp.reshape(logits, (n_edge, n_channel)), compute_dtype)
    edge_env_1d = xp.astype(xp.reshape(edge_env, (n_edge,)), compute_dtype)
    edge_positive = edge_env_1d > 0.0
    ones = xp.ones((n_edge,), dtype=compute_dtype, device=device)
    log_weight = 2.0 * xp.log(xp.where(edge_positive, edge_env_1d, ones))
    active = edge_positive
    if src_weight is not None:
        source_weight = xp.astype(xp.reshape(src_weight, (n_edge,)), compute_dtype)
        source_positive = source_weight > 0.0
        log_weight = log_weight + xp.log(xp.where(source_positive, source_weight, ones))
        active = active & source_positive
    if edge_mask is not None:
        mask = xp.astype(xp.reshape(edge_mask, (n_edge,)), compute_dtype)
        mask_positive = mask > 0.0
        # ``edge_mask`` is a binary validity mask, so its positive branch has
        # log-factor zero and only needs to participate in the active predicate.
        active = active & mask_positive

    effective_logits = logits_2d + log_weight[:, None]
    minus_inf = xp.full(
        (n_edge, n_channel),
        float("-inf"),
        dtype=compute_dtype,
        device=device,
    )
    effective_logits = xp.where(
        active[:, None],
        effective_logits,
        minus_inf,
    )
    null_mass = xp.reshape(
        softplus_t(xp.astype(z_bias_raw, compute_dtype)) + float(eps),
        (1, n_channel),
    )
    null_logit = xp.log(null_mass)

    # === Step 2. Destination-wise max including the physical null mass ===
    # The null initialization keeps empty and all-masked segments finite.
    group_max = xp_maximum_at(
        xp.zeros((n_nodes, n_channel), dtype=compute_dtype, device=device) + null_logit,
        dst,
        effective_logits,
    )  # (N, n_channel)
    edge_max = xp.take(group_max, dst, axis=0)

    # === Step 3. Normalize edge and null masses in the shared shifted frame ===
    edge_exp = xp.exp(effective_logits - edge_max)
    denom_sum = xp_add_at(
        xp.zeros((n_nodes, n_channel), dtype=compute_dtype, device=device),
        dst,
        edge_exp,
    )  # (N, n_channel)
    denominator = denom_sum + xp.exp(null_logit - group_max)
    alpha = edge_exp / xp.take(denominator, dst, axis=0)
    if promote:
        alpha = xp.astype(alpha, input_dtype)
    return xp.reshape(alpha, (n_edge, n_focus, n_head))
