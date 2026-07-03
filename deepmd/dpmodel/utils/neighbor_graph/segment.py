# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mask-aware, backend-dispatched segment reductions (the dpmodel scatter
primitive). Built on deepmd.dpmodel.array_api.xp_add_at so they work for
numpy / jax / torch. segment_index must be int64 (torch index_add requirement).
"""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
    xp_maximum_at,
)


def segment_sum(data: Array, segment_ids: Array, num_segments: int) -> Array:
    """out[s] = sum of data[i] over i with segment_ids[i] == s. Shape
    ``(num_segments, *data.shape[1:])``; empty segments are zero.
    """
    xp = array_api_compat.array_namespace(data)
    out = xp.zeros(
        (num_segments, *tuple(data.shape[1:])),
        dtype=data.dtype,
        device=array_api_compat.device(data),
    )
    return xp_add_at(out, segment_ids, data)


def segment_mean(data: Array, segment_ids: Array, num_segments: int) -> Array:
    """Per-segment mean; empty segments are zero (no division by zero)."""
    xp = array_api_compat.array_namespace(data)
    summed = segment_sum(data, segment_ids, num_segments)
    ones = xp.ones(
        (data.shape[0],), dtype=data.dtype, device=array_api_compat.device(data)
    )
    counts = segment_sum(ones[:, None], segment_ids, num_segments)  # (num_segments, 1)
    safe = xp.where(counts == 0, xp.ones_like(counts), counts)
    # broadcast counts over the trailing dims of summed
    shape = (num_segments,) + (1,) * (summed.ndim - 1)
    return summed / xp.reshape(safe, shape)


def segment_max(data: Array, segment_ids: Array, num_segments: int) -> Array:
    """out[s] = max of data[i] over i with segment_ids[i] == s.

    Shape ``(num_segments, *data.shape[1:])``; empty segments are ``-inf``
    (neutral element — callers guard with masks before consuming them).
    """
    xp = array_api_compat.array_namespace(data)
    out = xp.full(
        (num_segments, *tuple(data.shape[1:])),
        -xp.inf,
        dtype=data.dtype,
        device=array_api_compat.device(data),
    )
    return xp_maximum_at(out, segment_ids, data)


def segment_softmax(
    data: Array,
    segment_ids: Array,
    num_segments: int,
    mask: Array | None = None,
) -> Array:
    """Softmax over entries sharing a segment id, numerically stable.

    Mirrors the dense ``np_softmax`` max-subtraction trick with a PER-SEGMENT
    max. ``mask`` (bool, per entry) removes masked entries from the softmax
    entirely (zero weight AND excluded from the denominator). Empty or
    fully-masked segments produce all-zero weights (no NaN).
    """
    xp = array_api_compat.array_namespace(data)
    if mask is not None:
        # keep masked entries out of the per-segment max: send them to -inf
        neg = xp.full_like(data, -xp.inf)
        data_for_max = xp.where(mask, data, neg)
    else:
        data_for_max = data
    seg_max = segment_max(data_for_max, segment_ids, num_segments)
    # guard -inf (empty / fully-masked segments) so gather doesn't yield inf-inf
    seg_max = xp.where(xp.isinf(seg_max), xp.zeros_like(seg_max), seg_max)
    # shift data_for_max (masked entries already -inf), NOT the raw data:
    # a masked entry whose raw value exceeds the unmasked per-segment max by
    # more than the exp overflow threshold (~709 fp64 / ~88 fp32) would give
    # exp(+big) = inf, and the post-hoc inf * 0 mask multiply = nan, poisoning
    # the WHOLE segment through the denominator. exp(-inf) = 0 exactly.
    shifted = data_for_max - xp.take(seg_max, segment_ids, axis=0)
    ex = xp.exp(shifted)
    if mask is not None:
        # defensive no-op after the -inf shift (exp(-inf) == 0); kept so the
        # zero-weight guarantee never depends on the shift implementation
        ex = ex * xp.astype(mask, ex.dtype)
    denom = segment_sum(ex, segment_ids, num_segments)
    denom_e = xp.take(denom, segment_ids, axis=0)
    safe = xp.where(denom_e > 0, denom_e, xp.ones_like(denom_e))
    return ex / safe
