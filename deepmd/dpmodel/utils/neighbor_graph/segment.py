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


def _slot_occupancy(
    slot_weight: Array,
    segment_ids: Array,
    num_segments: int,
    capacity: Array,
) -> Array:
    """Per-entry fractional slot occupancy by capped water-filling.

    Solves, independently per segment, ``theta_j = min(1, w_j / lam)`` with
    the smallest ``lam >= 0`` such that ``sum_j theta_j <= capacity`` — the
    projection of the weights onto the capped simplex. When the capacity is
    not binding (``count(w_j > 0) <= capacity``) the water level is
    ``lam = 0`` and every positive-weight entry gets ``theta_j == 1``
    BITWISE (``w / max(w, 0) == w / w``); zero-weight entries get 0.

    Parameters
    ----------
    slot_weight
        Non-negative per-entry weights (the smooth cutoff envelope ``sw``),
        with shape ``(n,)``.
    segment_ids
        Segment id of each entry (int64), with shape ``(n,)``.
    num_segments
        Number of segments.
    capacity
        Per-segment slot capacity (``sel``), with shape ``(num_segments,)``.

    Returns
    -------
    theta : Array
        Per-entry occupancy in ``[0, 1]``, with shape ``(n,)``. Continuous in
        ``slot_weight`` (piecewise smooth), with ``theta_j -> 0`` as
        ``w_j -> 0`` whenever the segment's capacity is binding.
    """
    xp = array_api_compat.array_namespace(slot_weight)
    dev = array_api_compat.device(slot_weight)
    n = slot_weight.shape[0]
    # NOTE: no ``n == 0`` early-out -- the entry count is a data-dependent
    # (unbacked) symbol under torch.export and a Python branch on it raises
    # GuardOnDataDependentSymNode; every op below is empty-safe as-is.
    # sort by (segment asc, weight desc): stable argsort twice
    order_w = xp.argsort(slot_weight, descending=True, stable=True)
    order = xp.take(
        order_w,
        xp.argsort(xp.take(segment_ids, order_w, axis=0), stable=True),
        axis=0,
    )
    sw_s = xp.take(slot_weight, order, axis=0)
    seg_s = xp.take(segment_ids, order, axis=0)
    ones = xp.ones((n,), dtype=slot_weight.dtype, device=dev)
    counts = segment_sum(ones, segment_ids, num_segments)  # (S,)
    total = segment_sum(slot_weight, segment_ids, num_segments)  # (S,)
    # within-segment 1-based rank of each sorted entry
    offsets = xp.cumulative_sum(counts) - counts  # exclusive prefix
    iota = xp.cumulative_sum(ones) - 1.0  # arange(n) as float
    rank = iota - xp.take(offsets, seg_s, axis=0) + 1.0  # (n,) 1-based
    # suffix weight below rank k: T_k = total - (top-k prefix).  This cumsum
    # feeds ONLY the ``valid`` comparison below, so no gradient ever flows
    # through it -- deliberately: the entry axis is a data-dependent
    # (unbacked) size under torch.export, and a gradient-carrying cumsum
    # there guards ``numel <= 1`` in its backward (and padding workarounds
    # trip inductor's unbacked_bindings on the slice).  The differentiable
    # water mass ``t_star`` is recomputed via segment_sum instead.
    prefix = xp.cumulative_sum(sw_s)
    seg_prefix_off = xp.take(xp.cumulative_sum(total) - total, seg_s, axis=0)
    t_k = xp.take(total, seg_s, axis=0) - (prefix - seg_prefix_off)  # (n,)
    cap_e = xp.take(capacity, seg_s, axis=0)  # (n,)
    # cut k (top-k entries saturated at 1) is feasible iff the water level
    # lam_k = T_k / (cap - k) does not exceed the k-th largest weight
    room = cap_e - rank
    valid = xp.logical_and(room > 0.0, sw_s * room >= t_k)
    kstar = segment_max(xp.where(valid, rank, xp.zeros_like(rank)), seg_s, num_segments)
    kstar = xp.maximum(kstar, xp.zeros_like(kstar))  # empty segments: -inf -> 0
    # water mass at the chosen cut: the total weight of the UNSATURATED
    # entries (rank > kstar; equals total when kstar == 0).  Gradient-safe
    # rebuild of T_{k*}: index_add of a masked term, no scan involved.
    unsat = xp.astype(rank > xp.take(kstar, seg_s, axis=0), sw_s.dtype)
    t_star = segment_sum(sw_s * unsat, seg_s, num_segments)
    # cap - kstar >= 1 whenever a feasible cut exists (room > 0); a
    # non-positive capacity has no slots at all -> lam = inf -> theta = 0
    den2 = capacity - kstar
    has_room = den2 > 0.0
    lam = xp.where(
        has_room,
        t_star / xp.where(has_room, den2, xp.ones_like(den2)),
        xp.full_like(den2, xp.inf),
    )
    lam_e = xp.take(lam, segment_ids, axis=0)
    # theta = w / max(w, lam); safe-where the w == lam == 0 case BEFORE the
    # division (0/0 backward would poison gradients even at zero cotangent)
    den = xp.maximum(slot_weight, lam_e)
    pos = den > 0.0
    den_safe = xp.where(pos, den, xp.ones_like(den))
    return xp.where(pos, slot_weight / den_safe, xp.zeros_like(den))


def segment_softmax(
    data: Array,
    segment_ids: Array,
    num_segments: int,
    mask: Array | None = None,
    phantom_count: Array | None = None,
    phantom_logit: float = 0.0,
    slot_weight: Array | None = None,
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
    slot_weight
        Per-entry smooth slot weight (the cutoff envelope ``sw``), with
        shape ``(n,)``. REQUIRED whenever ``phantom_count`` is given: it
        drives the soft slot occupancy that keeps the phantom denominator
        strictly positive and cutoff-continuous (see Notes). Must vanish
        exactly when the entry's logit reaches ``phantom_logit`` at the
        cutoff (the smooth-envelope invariant of every caller).

    Returns
    -------
    weights : Array
        Per-entry softmax weights, with shape ``(n, *f)``.

    Raises
    ------
    ValueError
        If ``phantom_count`` is given without ``slot_weight``.

    Notes
    -----
    The phantom machinery reproduces the dense smooth-attention convention
    -- a fixed ``sel``-width softmax whose padding slots each hold
    ``exp(-attnw_shift)`` -- on a ragged edge set whose entry count varies
    with geometry, via SOFT SLOT OCCUPANCY: each entry occupies
    ``theta_j = min(1, sw_j / lam)`` of the segment's ``sel`` slots (capped
    water-filling of the envelope weights, so ``sum theta <= sel``) and
    contributes ``theta_j e_j + (1 - theta_j) relu(e_j - P)`` to the
    denominator, with the unoccupied capacity contributing
    ``relu(sel - sum theta) * P`` (``e_j = exp(l_j)``,
    ``P = exp(phantom_logit)``). Properties:

    - ``n_real <= sel``: every ``theta == 1`` bitwise, giving the EXACT
      dense fixed-width denominator ``sum e_j + (sel - n) P`` term for
      term, for arbitrary logits -- including logits below
      ``phantom_logit``.
    - in-design (all logits >= ``phantom_logit``): the relu is the identity
      and the total telescopes to the signed
      ``sum e_j - (n - sel) P``, independent of ``theta`` -- the sel-free
      carry-all compensation whose per-entry contribution vanishes at the
      cutoff (the boundary logit equals ``phantom_logit``), keeping the
      weights continuous when an entry enters or leaves for ARBITRARY
      degree.
    - strictly positive for any finite logits: every term is non-negative
      and the occupied mass is positive -- the raw signed denominator's
      zero crossing (reachable for negative counts because the envelope
      maps ``raw < -attnw_shift`` to ``l < phantom_logit`` at ``sw > 0``)
      cannot occur; no floor term is needed.
    - cutoff-continuous across the count 0 -> -1 transition for arbitrary
      logits: the entering entry has BOTH ``e -> P`` and ``theta -> 0``, so
      its bracket vanishes; per-entry weights are bounded by
      ``1 / theta_j`` (at most ``n_real / sel``-scale), and the caller's
      post-softmax ``sw`` factor keeps the boundary entry's own
      contribution vanishing.
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
    if ph_b is not None:
        if slot_weight is None:
            raise ValueError(
                "segment_softmax: phantom_count requires slot_weight (the "
                "smooth per-entry envelope) -- the sel-slot occupancy that "
                "keeps the phantom denominator positive and cutoff-continuous "
                "cannot be derived from the logits alone"
            )
        # exponent clamped at 80 (exp(80) ~ 5.5e34 is finite in BOTH float32
        # and float64): pl - seg_max > 80 means every real logit sits > 80
        # below the phantom logit; an unclamped exp would overflow (inf in
        # float32 already at ~88) and poison the sum with inf - inf.
        ph_arg = xp.minimum(
            xp.full_like(seg_max, phantom_logit) - seg_max,
            xp.full_like(seg_max, 80.0),
        )
        ph_term = xp.exp(ph_arg)
        # SOFT SLOT OCCUPANCY: real logits are not bounded below by
        # ``phantom_logit`` (the smooth envelope maps ``raw < -shift`` to
        # ``l < -shift`` at ``sw > 0``), so the plain SIGNED denominator
        # ``sum_j e_j + (sel - n) * P`` crosses zero for negative counts --
        # a pole reachable with finite, valid model parameters.  Instead,
        # each entry occupies ``theta_j`` of the segment's ``sel`` dense
        # slots (``theta`` = capped water-filling of the envelope weights,
        # see :func:`_slot_occupancy`) and contributes
        #
        #     theta_j * e_j + (1 - theta_j) * relu(e_j - P)
        #
        # while the unoccupied capacity contributes
        # ``relu(sel - sum theta) * P``.  Properties:
        #
        # - ``n <= sel``: theta == 1 BITWISE (water level 0) -> every
        #   bracket is exactly ``e_j`` and the remainder is exactly
        #   ``(sel - n) * P``: the dense fixed-width softmax, term for
        #   term, for ARBITRARY logits (including below ``phantom_logit``).
        # - in-design (all ``l_j >= phantom_logit``): the relu is the
        #   identity and the total telescopes to the signed
        #   ``sum e_j - (n - sel) * P`` INDEPENDENT of theta.
        # - strictly positive: every term is non-negative and the occupied
        #   mass ``sum theta_j e_j > 0`` whenever any theta > 0 -- no pole
        #   for any finite logits, no floor needed.
        # - cutoff-continuous: an edge at its boundary has ``e -> P`` AND
        #   ``theta -> 0`` (its envelope weight vanishes), so its bracket
        #   ``theta * e + (1 - theta) * relu(e - P) -> 0`` and the freed
        #   capacity term picks up exactly nothing; per-entry weights are
        #   bounded by ``1 / theta_j`` and their post-softmax ``sw``
        #   factors keep the boundary contribution vanishing.
        if mask is not None:
            m_f = xp.astype(mask, ex.dtype)
            n_real_seg = segment_sum(m_f, segment_ids, num_segments)
            sw_eff = xp.astype(slot_weight, ex.dtype) * m_f
        else:
            n_real_seg = segment_sum(
                xp.ones(data.shape[:1], dtype=ex.dtype, device=dev),
                segment_ids,
                num_segments,
            )
            sw_eff = xp.astype(slot_weight, ex.dtype)
        cap = n_real_seg + xp.astype(
            xp.reshape(phantom_count, (-1,)), ex.dtype
        )  # (S,) == sel
        theta = _slot_occupancy(sw_eff, segment_ids, num_segments, cap)
        theta_b = xp.reshape(theta, theta.shape + (1,) * (ex.ndim - 1))
        ph_e = xp.take(ph_term, segment_ids, axis=0)  # (n, *f)
        excess = xp.maximum(ex - ph_e, xp.zeros_like(ex))
        bracket = theta_b * ex + (1.0 - theta_b) * excess
        denom = segment_sum(bracket, segment_ids, num_segments)
        occ = segment_sum(theta, segment_ids, num_segments)  # (S,)
        free = xp.maximum(cap - occ, xp.zeros_like(cap))
        denom = denom + xp.reshape(free, free.shape + (1,) * (denom.ndim - 1)) * ph_term
    else:
        denom = segment_sum(ex, segment_ids, num_segments)
    denom_e = xp.take(denom, segment_ids, axis=0)
    safe = xp.where(denom_e > 0, denom_e, xp.ones_like(denom_e))
    return ex / safe
