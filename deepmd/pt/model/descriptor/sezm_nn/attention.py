# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Attention utilities for SeZM message passing.

This module implements the destination-wise envelope-gated softmax used by the
SO(2) attention path in the SeZM descriptor.
"""

from __future__ import (
    annotations,
)

import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def segment_envelope_gated_softmax(
    logits: torch.Tensor,
    edge_env: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
    z_bias_raw: torch.Tensor,
    eps: float,
    src_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute destination-wise envelope-gated softmax attention.

    Parameters
    ----------
    logits
        Attention logits with shape (E, F, H).
    edge_env
        Cutoff envelope weights with shape (E, 1) or (E,).
    dst
        Destination node indices with shape (E,).
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

    Returns
    -------
    torch.Tensor
        Normalized edge weights with shape (E, F, H).
    """
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    input_dtype = logits.dtype
    compute_dtype = (
        torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
    )

    # === Step 1. Build factor-wise effective logits ===
    # Computing the logarithms before multiplying the factors avoids losing a
    # physically nonzero edge when ``edge_env**2 * src_weight`` underflows.
    logits_2d = logits.reshape(n_edge, n_channel).to(dtype=compute_dtype)
    edge_env_1d = edge_env.reshape(n_edge).to(dtype=compute_dtype)
    edge_positive = edge_env_1d > 0.0
    ones = torch.ones_like(edge_env_1d)
    log_weight = 2.0 * torch.log(torch.where(edge_positive, edge_env_1d, ones))
    active = edge_positive
    if src_weight is not None:
        source_weight = src_weight.reshape(n_edge).to(dtype=compute_dtype)
        source_positive = source_weight > 0.0
        log_weight = log_weight + torch.log(
            torch.where(source_positive, source_weight, ones)
        )
        active = active & source_positive
    effective_logits = torch.where(
        active.reshape(n_edge, 1),
        logits_2d + log_weight.reshape(n_edge, 1),
        torch.full_like(logits_2d, float("-inf")),
    )

    null_mass = (F.softplus(z_bias_raw.to(dtype=compute_dtype)) + float(eps)).reshape(
        1, n_channel
    )
    null_logit = torch.log(null_mass)
    dst_index = dst.reshape(n_edge, 1).expand(n_edge, n_channel)

    # === Step 2. Destination-wise max including the physical null mass ===
    # Initializing every segment with ``null_logit`` keeps empty and all-masked
    # segments finite without a separate fallback branch.
    group_max = null_logit.expand(n_nodes, n_channel).clone()
    group_max = torch.scatter_reduce(
        group_max,
        0,
        dst_index,
        effective_logits,
        reduce="amax",
        include_self=True,
    )
    edge_max = group_max.index_select(0, dst)

    # === Step 3. Normalize edge and null masses in the shared shifted frame ===
    edge_exp = torch.exp(effective_logits - edge_max)
    denom_sum = torch.zeros(
        n_nodes,
        n_channel,
        dtype=compute_dtype,
        device=logits.device,
    )
    denom_sum = torch.scatter_add(denom_sum, 0, dst_index, edge_exp)
    denominator = denom_sum + torch.exp(null_logit - group_max)
    alpha = edge_exp / denominator.index_select(0, dst)

    if alpha.dtype != input_dtype:
        alpha = alpha.to(dtype=input_dtype)
    return alpha.reshape(n_edge, n_focus, n_head)
