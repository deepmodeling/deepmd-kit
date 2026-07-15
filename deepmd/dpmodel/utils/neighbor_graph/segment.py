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
    phantom_count: Array | None = None,
    phantom_logit: float = 0.0,
) -> Array:
    """Softmax over entries sharing a segment id, numerically stable.

    Mirrors the dense ``np_softmax`` max-subtraction trick with a PER-SEGMENT
    max. Empty or fully-masked segments produce all-zero weights (no NaN).

    Parameters
    ----------
    data
        Per-entry logits, with shape ``(n, *f)``.
    segment_ids
        Segment id of each entry (int64), with shape ``(n,)``.
    num_segments
        Number of segments.
    mask
        Optional per-entry bool mask, with shape ``(n,)``. Masked entries are
        removed from the softmax entirely (zero weight AND excluded from the
        denominator).
    phantom_count
        Optional per-segment SIGNED count of virtual entries, with shape
        ``(num_segments,)``. Each unit adds (or, when negative, removes) one
        ``exp(phantom_logit)`` term to the segment's DENOMINATOR only (no
        output rows are produced for them). Negative counts express the
        sel-free carry-all convention beyond ``sel`` -- see Notes.
    phantom_logit
        The raw logit of every phantom entry.

    Returns
    -------
    weights : Array
        Per-entry softmax weights, with shape ``(n, *f)``.

    Notes
    -----
    The phantom entries reproduce the dense smooth-attention convention -- a
    fixed ``sel``-width softmax whose padding slots each hold
    ``exp(-attnw_shift)`` -- on a ragged edge set whose entry count varies
    with geometry: passing the SIGNED ``phantom_count = sel - n_real`` makes
    the denominator ``sum_j exp(l_j) + (sel - n_real) * exp(phantom_logit)``
    == ``sum_j (exp(l_j) - exp(phantom_logit)) + sel * exp(phantom_logit)``:
    every entry's net denominator contribution vanishes exactly at the
    cutoff (its boundary logit equals ``phantom_logit`` by the smooth
    envelope), so the attention weights are continuous when an entry enters
    or leaves the segment for ARBITRARY degree -- including the sel-free
    carry-all regime ``n_real > sel``, where a clamped (non-negative) count
    would drop the compensation and leave a finite ``exp(-attnw_shift)``
    denominator step at the crossing.  For ``n_real <= sel`` the scheme is
    term-for-term equal to the dense softmax.  Real logits are NOT bounded
    below by ``phantom_logit`` (the smooth envelope maps pre-shift logits
    ``raw < -attnw_shift`` to ``l < phantom_logit`` at ``sw > 0``), so for
    NEGATIVE counts the raw signed denominator can cross zero; a smooth
    strictly-positive floor (see the inline comment at the denominator)
    keeps the normalization finite and positive there.  The floor is gated
    to the negative-count, small-denominator regime, so for
    ``phantom_count >= 0`` (a plain positive softmax sum -- no pole exists)
    and for every in-design negative-count segment the output is
    BIT-IDENTICAL to the floor-free formula: the dense term-for-term parity
    is exact, not merely approximate.
    """
    xp = array_api_compat.array_namespace(data)
    dev = array_api_compat.device(data)
    if mask is not None:
        # broadcast mask (n,) over any trailing feature dims of data (n, *f)
        mask_b = xp.reshape(mask, mask.shape + (1,) * (data.ndim - 1))
        # keep masked entries out of the per-segment max: send them to -inf
        neg = xp.full_like(data, -xp.inf)
        data_for_max = xp.where(mask_b, data, neg)
    else:
        data_for_max = data
    seg_max = segment_max(data_for_max, segment_ids, num_segments)
    ph_b = None
    if phantom_count is not None:
        # phantom entries participate in the max exactly like dense's padding
        # slots do in np_softmax's row max (dense: m = max(real, -shift));
        # this also guards exp-overflow when every real logit < phantom_logit.
        ph_b = xp.reshape(
            phantom_count, phantom_count.shape + (1,) * (seg_max.ndim - 1)
        )
        seg_max = xp.where(
            ph_b > 0,
            xp.maximum(seg_max, xp.full_like(seg_max, phantom_logit)),
            seg_max,
        )
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
        ex = ex * xp.astype(mask_b, ex.dtype)
    denom = segment_sum(ex, segment_ids, num_segments)
    if ph_b is not None:
        # exponent clamped at 80 (exp(80) ~ 5.5e34 is finite in BOTH float32
        # and float64): pl - seg_max > 80 means every real logit sits > 80
        # below the phantom logit; an unclamped exp would overflow (inf in
        # float32 already at ~88) and poison the signed sum with inf - inf.
        ph_arg = xp.minimum(
            xp.full_like(seg_max, phantom_logit) - seg_max,
            xp.full_like(seg_max, 80.0),
        )
        ph_term = xp.exp(ph_arg)
        ph_f = xp.astype(ph_b, denom.dtype)
        denom = denom + ph_f * ph_term
        # Smooth strictly-positive floor for NEGATIVE counts only: real
        # logits are not bounded below by ``phantom_logit`` (the smooth
        # envelope maps ``raw < -shift`` to ``l < -shift`` at ``sw > 0``),
        # so with a negative count the signed denominator D can cross zero
        # -- a pole in the attention weights reachable with finite, valid
        # model parameters.  Exponential-tail floor:
        #
        #     D~ = D + delta * exp(-D / delta),   delta = sel*ph_term / 40
        #
        # applied ONLY where (a) the count is negative (for count >= 0 the
        # denominator is a plain positive softmax sum -- no pole -- and the
        # dense term-for-term parity must stay BIT-exact) and (b)
        # D < 40 * delta (beyond that the correction is <= exp(-40) * delta
        # < 1e-19 RELATIVE to D -- sub-ulp in float32 AND float64, so the
        # gate boundary is bit-invisible; in-design every real logit is
        # >= phantom_logit, hence D >= sel * ph_term == 40 * delta and the
        # floor never fires).  Properties inside the active region: C-inf
        # in the logits, D~ >= delta > 0 for any finite logits, weights
        # bounded by ~40 * n_real / sel, and the boundary-edge cancellation
        # survives (a smooth function of the already-continuous D).
        #
        # The inactive branch substitutes (0, 1) for (D, delta) BEFORE the
        # division: even a zero cotangent cannot rescue exp(-D/delta) under
        # autodiff when delta underflows (backward forms D/delta**2 -> inf,
        # and 0 * inf = NaN in torch/jax float32), so the pathological
        # division must never be constructed.  The exponent cap 80 bounds
        # the active branch's exp for float32 while keeping D~ positive for
        # any physical edge count (needs n_real/sel < exp(80)/40 ~ 1e33).
        if mask is not None:
            n_real_seg = segment_sum(
                xp.astype(mask, denom.dtype), segment_ids, num_segments
            )
        else:
            n_real_seg = segment_sum(
                xp.ones(data.shape[:1], dtype=denom.dtype, device=dev),
                segment_ids,
                num_segments,
            )
        sel_b = (
            xp.reshape(n_real_seg, n_real_seg.shape + (1,) * (denom.ndim - 1)) + ph_f
        )
        delta = xp.maximum(sel_b, xp.ones_like(sel_b)) * ph_term / 40.0
        active = xp.logical_and(ph_f < 0.0, denom < 40.0 * delta)
        denom_a = xp.where(active, denom, xp.zeros_like(denom))
        delta_a = xp.where(active, delta, xp.ones_like(delta))
        e_arg = xp.minimum(-denom_a / delta_a, xp.full_like(denom, 80.0))
        denom = xp.where(active, denom + delta_a * xp.exp(e_arg), denom)
    denom_e = xp.take(denom, segment_ids, axis=0)
    safe = xp.where(denom_e > 0, denom_e, xp.ones_like(denom_e))
    return ex / safe
