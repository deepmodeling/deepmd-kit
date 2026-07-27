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
    """Per-entry fractional slot occupancy by C1 capped water-filling.

    Solves, independently per segment, ``theta_j = h(w_j * u)`` with the
    C1 plateau saturator ``h(t) = t * (2 - t)`` for ``t < 1`` and
    ``h(t) = 1`` for ``t >= 1``, and the largest inverse water level
    ``u >= 0`` such that ``sum_j theta_j <= capacity``.  Because
    ``h'(1) = 0``, both the per-entry saturation boundary and the
    water-filling active-set transitions are C1 in the weights -- the
    min-based projection's kinks are exactly what this form removes.

    When the capacity is not binding (``count(w_j > 0) <= capacity``)
    every positive-weight entry gets ``theta_j == 1`` BITWISE (a literal
    1.0 through the plateau branch); zero-weight entries get 0.

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
        Per-entry occupancy in ``[0, 1]``, with shape ``(n,)``. C1 in
        ``slot_weight``, with ``theta_j -> 0`` as ``w_j -> 0`` whenever the
        segment's capacity is binding.

    Notes
    -----
    For a cut with ``k`` saturated (plateau) entries the water equation
    ``k + sum_unsat (2 w u - w^2 u^2) = capacity`` is quadratic in ``u``;
    the physical (smaller) root in cancellation-free form is
    ``u = (capacity - k) / (S + sqrt(S^2 - Q (capacity - k)))`` with
    ``S = sum_unsat w`` and ``Q = sum_unsat w^2``.  The per-rank suffix
    sums that SELECT the cut feed only comparisons (no gradient -- a
    gradient-carrying cumsum on the data-dependent entry axis breaks
    torch.export, see the inline comment), while the differentiable
    ``S``/``Q`` at the chosen cut are rebuilt as masked ``segment_sum``.
    """
    xp = array_api_compat.array_namespace(slot_weight)
    dev = array_api_compat.device(slot_weight)
    n = slot_weight.shape[0]
    dt = slot_weight.dtype
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
    ones = xp.ones((n,), dtype=dt, device=dev)
    n_pos = segment_sum(
        xp.astype(slot_weight > 0.0, dt), segment_ids, num_segments
    )  # (S,)
    # The ENTIRE cut selection below runs in float64: the suffix moments are
    # formed as ``total - prefix`` and the discriminant as ``S^2 - Q*room``,
    # both ill-conditioned subtractions -- in float32, cutoff weights
    # spanning ordinary decades (e.g. [1, 0.1, 0.01, 5e-7]) lose the small
    # suffixes entirely, the valid saturated cut is rejected, and the
    # fallback root violates the capacity equation by O(1).  The selection
    # feeds ONLY comparisons (kstar), so the upcast costs no gradient and no
    # export surface; the differentiable quantities at the chosen cut are
    # rebuilt in the compute dtype from positive-only masked sums.
    f64 = xp.float64
    sw64 = xp.astype(slot_weight, f64)
    ones64 = xp.ones((n,), dtype=f64, device=dev)
    sw64_s = xp.take(sw64, order, axis=0)
    counts64 = segment_sum(ones64, segment_ids, num_segments)  # (S,)
    total_s = segment_sum(sw64, segment_ids, num_segments)  # (S,)
    total_q = segment_sum(sw64 * sw64, segment_ids, num_segments)
    # within-segment 1-based rank of each sorted entry
    offsets = xp.cumulative_sum(counts64) - counts64  # exclusive prefix
    iota = xp.cumulative_sum(ones64) - 1.0  # arange(n) as float
    rank = iota - xp.take(offsets, seg_s, axis=0) + 1.0  # (n,) 1-based
    # Per-rank suffix sums S_k / Q_k (weights strictly below rank k).  These
    # cumsums feed ONLY the cut-selection comparisons below, so no gradient
    # ever flows through them -- deliberately: the entry axis is a
    # data-dependent (unbacked) size under torch.export, and a
    # gradient-carrying cumsum there guards ``numel <= 1`` in its backward
    # (and padding workarounds trip inductor's unbacked_bindings on the
    # slice).  The differentiable sums at the chosen cut are rebuilt as
    # masked segment_sum below.
    prefix_s = xp.cumulative_sum(sw64_s)
    prefix_q = xp.cumulative_sum(sw64_s * sw64_s)
    off_s = xp.take(xp.cumulative_sum(total_s) - total_s, seg_s, axis=0)
    off_q = xp.take(xp.cumulative_sum(total_q) - total_q, seg_s, axis=0)
    s_k = xp.take(total_s, seg_s, axis=0) - (prefix_s - off_s)  # (n,)
    q_k = xp.take(total_q, seg_s, axis=0) - (prefix_q - off_q)  # (n,)
    cap_e = xp.astype(xp.take(capacity, seg_s, axis=0), f64)  # (n,)
    room = cap_e - rank
    disc_k = s_k * s_k - q_k * room
    disc_pos = disc_k > 0.0
    sq_k = xp.where(disc_pos, xp.sqrt(xp.where(disc_pos, disc_k, ones64)), 0.0 * ones64)
    den_k = s_k + sq_k
    den_k_pos = den_k > 0.0
    u_k = xp.where(
        den_k_pos, room / xp.where(den_k_pos, den_k, ones64), xp.zeros_like(room)
    )
    # cut k (top-k entries saturated) is feasible iff a real water level
    # exists (disc >= 0) and the k-th largest weight is still saturated
    # (w_k * u_k >= 1); taking the LARGEST feasible k also puts every
    # unsaturated weight below the plateau (verified against a bisection
    # reference in the unit tests)
    valid = xp.logical_and(
        xp.logical_and(room > 0.0, disc_k >= 0.0), sw64_s * u_k >= 1.0
    )
    kstar64 = segment_max(
        xp.where(valid, rank, xp.zeros_like(rank)), seg_s, num_segments
    )
    kstar64 = xp.maximum(kstar64, xp.zeros_like(kstar64))  # empty: -inf -> 0
    kstar = xp.astype(kstar64, dt)  # small integer, exact in any dtype
    # differentiable S/Q of the unsaturated set (rank > kstar; the whole
    # segment when kstar == 0): masked index_add of POSITIVE terms -- no
    # cancellation, safe in the compute dtype
    unsat = xp.astype(rank > xp.take(kstar64, seg_s, axis=0), dt)
    s_star = segment_sum(sw_s * unsat, seg_s, num_segments)
    q_star = segment_sum(sw_s * sw_s * unsat, seg_s, num_segments)
    cap_rem = capacity - kstar
    disc = s_star * s_star - q_star * cap_rem
    # safe-where every branch BEFORE the sqrt/division (an unselected
    # sqrt(0) backward is inf, and 0 * inf = NaN in float32 autodiff)
    dpos = disc > 0.0
    one_g = xp.ones_like(disc)
    sq = xp.where(dpos, xp.sqrt(xp.where(dpos, disc, one_g)), xp.zeros_like(disc))
    den_u = s_star + sq
    upos = xp.logical_and(den_u > 0.0, cap_rem > 0.0)
    u = xp.where(upos, cap_rem / xp.where(upos, den_u, one_g), xp.zeros_like(disc))
    # binding <=> even full saturation of every positive weight exceeds cap
    binding = n_pos > capacity
    t = slot_weight * xp.take(u, segment_ids, axis=0)
    theta_soft = xp.where(t >= 1.0, ones, t * (2.0 - t))
    return xp.where(
        xp.take(binding, segment_ids, axis=0),
        theta_soft,
        xp.astype(slot_weight > 0.0, dt),
    )


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
    ``theta_j`` of the segment's ``sel`` slots (C1 capped water-filling of
    the envelope weights, see :func:`_slot_occupancy`) and contributes the
    C1 bracket ``theta_j e_j + (1 - theta_j)(e_j - P) chi_j`` to the
    denominator (``chi = (e/P)**(1/theta)`` below ``P``, else 1), with the
    unoccupied capacity contributing ``relu(sel - sum theta) * P``
    (``e_j = exp(l_j)``, ``P = exp(phantom_logit)``). Properties:

    - ``n_real <= sel``: every ``theta == 1`` bitwise, giving the EXACT
      dense fixed-width denominator ``sum e_j + (sel - n) P`` term for
      term, for arbitrary logits -- including logits below
      ``phantom_logit``.
    - in-design (all logits >= ``phantom_logit``): ``chi == 1`` and the
      total telescopes to the signed
      ``sum e_j - (n - sel) P``, independent of ``theta`` -- the sel-free
      carry-all compensation whose per-entry contribution vanishes at the
      cutoff (the boundary logit equals ``phantom_logit``), keeping the
      weights continuous when an entry enters or leaves for ARBITRARY
      degree.
    - strictly positive for any finite logits: the bracket is bounded below
      by ``theta (1 - 1/e0) e`` -- the raw signed denominator's zero
      crossing (reachable for negative counts because the envelope maps
      ``raw < -attnw_shift`` to ``l < phantom_logit`` at ``sw > 0``) cannot
      occur; no floor term is needed.
    - C1 (first-derivative-continuous) in the logits and envelope: the
      damped tail meets the in-design branch at ``e == P`` with matching
      slope, and the occupancy's plateau saturator makes water-filling
      active-set changes kink-free -- weight GRADIENTS, hence forces, are
      continuous across the below-phantom surface (second derivatives
      still jump there: exact in-design linearity and smoothness beyond C1
      are mutually exclusive with bitwise dense parity).
    - cutoff-continuous across the count 0 -> -1 transition for arbitrary
      logits: the entering entry has BOTH ``e -> P`` and ``theta -> 0``, so
      its bracket vanishes; per-entry weights are bounded by
      ``(e0/(e0 - 1)) / theta_j`` (``n_real / sel``-scale), and the
      caller's post-softmax ``sw`` factor keeps the boundary entry's own
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
        # slots (theta = C1 capped water-filling of the envelope weights,
        # see :func:`_slot_occupancy`) and contributes the C1 bracket
        # built below.  Properties (details in the class docstring Notes):
        # bitwise dense for n <= sel; theta-independent signed formula
        # in-design; strictly positive; C1 in logits and envelope (forces
        # carry no kink at the below-phantom surface or at water-filling
        # active-set changes); cutoff-continuous at every crossing.
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
        # Per-entry bracket, C1 in the logit:
        #
        #   B = theta*e + (1 - theta) * (e - P) * chi,
        #   chi = (e/P)**(1/theta) for e < P, else 1
        #
        # For e >= P this is the exact linear in-design branch (telescopes
        # to the signed formula, theta-independent).  For e < P the
        # (e/P)**(1/theta) damping meets the linear branch at e == P with
        # value theta*P AND slope 1 (chi -> 1, (e-P)*chi' -> 0), so the
        # below-phantom surface carries no force kink (the relu form jumped
        # the slope theta*P -> P there).  The damped correction is bounded:
        # with a = e/P, factoring out e gives
        # B = e * [theta + (1-theta)(a-1)a**(1/theta - 1)], and on (0, 1)
        # the factor (a-1)a**(1/theta - 1) attains its minimum
        # -theta*(1-theta)**(1/theta - 1) at a = 1 - theta, so
        # B/e >= theta*(1 - (1-theta)**(1/theta)) >= theta*(1 - 1/e0)
        # (e0 = Euler's number; (1-theta)**(1/theta) increases to 1/e0 as
        # theta -> 0).  Hence B > 0.63*theta*e -- strictly positive with
        # per-entry weights bounded by ~1.6/theta.  theta == 1 is BITWISE
        # ``e`` with no special case: the (1 - theta) = 0 factor kills the
        # finite tail term exactly.  theta -> 0 or e -> 0 send the tail to
        # zero.
        #
        # chi is computed ENTIRELY IN LOG SPACE: log(e/P) is the already
        # available ``shifted - ph_arg``, so no log() and no division by
        # ``ph_e`` ever enter the autodiff graph.  This matters for
        # in-design HIGH logits (count >= 0): ``ph_e = exp(pl - m)`` goes
        # subnormal when the segment max is large, and a log(ex_t / ph_e)
        # formulation -- even with ex_t where-substituted to ph_e -- pushes
        # ``1 / ph_e`` (torch) or ``ph_e**2`` (jax) past the float32 range
        # in the UNSELECTED branch's backward, whose 0 * inf poisons the
        # selected gradient with NaN.  The log-ratio is clamped at -1e4
        # (exp(-1e4) == 0 in both precisions) so masked entries' -inf
        # shifted logits cannot reach the division either; the only division
        # is by the where-substituted theta.
        one_t = xp.ones_like(ex)
        log_ratio = xp.maximum(
            shifted - xp.take(ph_arg, segment_ids, axis=0),
            xp.full_like(ex, -1.0e4),
        )
        below = log_ratio < 0.0
        tail_on = xp.logical_and(below, theta_b > 0.0)
        lr_t = xp.where(tail_on, log_ratio, xp.zeros_like(ex))
        th_t = xp.where(tail_on, theta_b * one_t, one_t)
        chi = xp.where(tail_on, xp.exp(lr_t / th_t), one_t)
        # zero-occupancy entries contribute nothing below P
        dead = xp.logical_and(below, xp.logical_not(tail_on))
        chi = xp.where(dead, xp.zeros_like(chi), chi)
        theta_dead = xp.where(dead, xp.zeros_like(theta_b), theta_b * one_t)
        bracket = theta_dead * ex + (1.0 - theta_dead) * (ex - ph_e) * chi
        denom = segment_sum(bracket, segment_ids, num_segments)
        occ = segment_sum(theta, segment_ids, num_segments)  # (S,)
        free = xp.maximum(cap - occ, xp.zeros_like(cap))
        denom = denom + xp.reshape(free, free.shape + (1,) * (denom.ndim - 1)) * ph_term
    else:
        denom = segment_sum(ex, segment_ids, num_segments)
    denom_e = xp.take(denom, segment_ids, axis=0)
    safe = xp.where(denom_e > 0, denom_e, xp.ones_like(denom_e))
    return ex / safe
