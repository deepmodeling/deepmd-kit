# SPDX-License-Identifier: LGPL-3.0-or-later
"""segment_max / segment_softmax (NeighborGraph PR-D segment toolkit)."""

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    segment_max,
    segment_softmax,
)


class TestSegmentMax:
    def test_basic(self) -> None:
        data = np.array([1.0, 5.0, 2.0, -3.0])
        ids = np.array([0, 0, 2, 2], dtype=np.int64)
        out = segment_max(data, ids, 3)
        assert out[0] == 5.0
        assert np.isneginf(out[1])  # empty segment
        assert out[2] == 2.0

    def test_trailing_dims(self) -> None:
        data = np.array([[1.0, -2.0], [3.0, -4.0], [0.0, 9.0]])
        ids = np.array([1, 1, 0], dtype=np.int64)
        out = segment_max(data, ids, 2)
        np.testing.assert_allclose(out[0], [0.0, 9.0])
        np.testing.assert_allclose(out[1], [3.0, -2.0])

    def test_torch_matches_numpy(self) -> None:
        import torch

        data = np.array([0.3, 1.2, -0.7, 2.0])
        ids = np.array([0, 0, 1, 1], dtype=np.int64)
        ref = segment_max(data, ids, 2)
        out = segment_max(torch.from_numpy(data), torch.from_numpy(ids), 2)
        np.testing.assert_allclose(out.numpy(), ref)


class TestSegmentSoftmax:
    def test_matches_dense(self) -> None:
        logits = np.array([1.0, 2.0, 0.5, -1.0])
        ids = np.array([0, 0, 0, 1], dtype=np.int64)
        w = segment_softmax(logits, ids, 2)
        ref0 = np.exp(np.array([1.0, 2.0, 0.5]) - 2.0)
        ref0 = ref0 / ref0.sum()
        np.testing.assert_allclose(w[:3], ref0, atol=1e-12)
        np.testing.assert_allclose(w[3], 1.0, atol=1e-12)

    def test_stable_large_logits(self) -> None:
        logits = np.array([1e30, 1e30 + 1.0])
        ids = np.array([0, 0], dtype=np.int64)
        w = segment_softmax(logits, ids, 1)
        assert not np.any(np.isnan(w))
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_masked_entries_zero(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        ids = np.array([0, 0, 0], dtype=np.int64)
        mask = np.array([True, False, True])
        w = segment_softmax(logits, ids, 1, mask=mask)
        assert w[1] == 0.0
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
        # masked entry excluded from the denominator too
        ref = np.exp(np.array([1.0, 3.0]) - 3.0)
        ref = ref / ref.sum()
        np.testing.assert_allclose(w[[0, 2]], ref, atol=1e-12)

    def test_all_masked_segment_is_zero_no_nan(self) -> None:
        logits = np.array([1.0, 2.0, 5.0])
        ids = np.array([0, 0, 1], dtype=np.int64)
        mask = np.array([True, True, False])
        w = segment_softmax(logits, ids, 2, mask=mask)
        assert not np.any(np.isnan(w))
        assert w[2] == 0.0

    def test_empty_segment_no_nan(self) -> None:
        logits = np.array([1.0, 2.0])
        ids = np.array([0, 0], dtype=np.int64)
        w = segment_softmax(logits, ids, 3)
        assert not np.any(np.isnan(w))

    def test_torch_matches_numpy(self) -> None:
        import torch

        logits = np.array([0.3, 1.2, -0.7, 2.0])
        ids = np.array([0, 0, 1, 1], dtype=np.int64)
        mask = np.array([True, True, True, False])
        ref = segment_softmax(logits, ids, 2, mask=mask)
        out = segment_softmax(
            torch.from_numpy(logits),
            torch.from_numpy(ids),
            2,
            mask=torch.from_numpy(mask),
        )
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-12)


def test_masked_entry_larger_than_unmasked_max_no_nan() -> None:
    """A masked entry FAR ABOVE the unmasked max must not poison the segment.

    Regression (CodeRabbit #5715): shifting the raw data let a huge masked
    logit overflow exp() to inf, and inf * 0 (mask multiply) = nan summed into
    the denominator, contaminating every entry of the segment. The shift must
    use the masked (-inf) values so masked entries exp() to exactly zero.
    """
    data = np.array([1.0, 2.0, 1e5], dtype=np.float64)  # 1e5 - 2 >> 709
    ids = np.zeros(3, dtype=np.int64)
    mask = np.array([True, True, False])
    out = segment_softmax(data, ids, 1, mask=mask)
    assert np.all(np.isfinite(out))
    ref = np.exp([1.0, 2.0]) / np.exp([1.0, 2.0]).sum()
    np.testing.assert_allclose(out[:2], ref, rtol=1e-12)
    assert out[2] == 0.0


class TestSignedPhantomStrictPositivity:
    """The SIGNED phantom denominator must stay strictly positive for
    arbitrary finite logits (OutisLi / njzjz-bot review).

    Real logits are not bounded below by ``phantom_logit``: the smooth
    envelope maps pre-shift logits ``raw < -attnw_shift`` to
    ``l < phantom_logit`` whenever ``sw > 0``.  With a negative count the
    raw signed denominator ``sum exp(l) + (sel - n) exp(ph)`` then crosses
    zero (e.g. ``sel=1``, logits ``[-21, -21]``, ``ph=-20``), which used to
    hit the ``denom > 0`` where-guard and return weights ``[1, 1]`` (sum 2)
    on one side and a pole on the other.  The smooth floor
    ``(D + sqrt(D^2 + delta^2)) / 2`` keeps the normalization finite,
    positive and smooth for any finite logits.
    """

    def test_reviewer_repro_finite_positive(self) -> None:
        # sel=1, two real entries below the phantom logit -> raw D < 0.
        data = np.array([-21.0, -21.0])
        seg = np.array([0, 0])
        w = segment_softmax(
            data,
            seg,
            1,
            phantom_count=np.array([-1.0]),  # sel - n_real = 1 - 2
            phantom_logit=-20.0,
        )
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)
        # a positive normalization cannot return the where-guard's [1, 1]
        assert w.sum() < 2.0

    def test_njzjz_repro_three_entries(self) -> None:
        data = np.array([-100.0, -100.0, -100.0])
        seg = np.array([0, 0, 0])
        w = segment_softmax(
            data,
            seg,
            1,
            phantom_count=np.array([-2.0]),  # sel=1, n_real=3
            phantom_logit=-20.0,
        )
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)
        # deep-suppressed entries with a huge phantom deficit -> near-zero
        # weights, NOT the where-guard's [1, 1, 1].
        assert w.sum() < 1.0

    def test_smooth_across_former_zero_crossing(self) -> None:
        """Sweep a logit through the raw denominator's zero crossing: the
        weights must stay finite and BOUNDED (<= 40 * n_real / sel from the
        floor scale), and their increments must SCALE with the sweep
        resolution -- the smoothness signature a pole cannot fake (at a
        pole, refining the grid does not shrink the largest step).
        """

        def _sweep(lo: float, hi: float, num: int) -> np.ndarray:
            outs = []
            for l2 in np.linspace(lo, hi, num):
                w = segment_softmax(
                    np.array([-21.0, float(l2)]),
                    np.array([0, 0]),
                    1,
                    phantom_count=np.array([-1.0]),
                    phantom_logit=-20.0,
                )
                assert np.all(np.isfinite(w))
                assert np.all(w >= 0.0)
                # floor-scale bound: 40 * n_real / sel = 80 here
                assert np.all(w <= 80.0)
                outs.append(w)
            return np.stack(outs)

        # D_raw(l) = exp(-21) + exp(l) - exp(-20) crosses zero at l = ln(
        # exp(-20) - exp(-21)) ~= -20.4587 (the LocalAtten sw-sweep pole
        # from the review, expressed directly in logit space).
        crossing = np.log(np.exp(-20.0) - np.exp(-21.0))
        coarse = _sweep(crossing - 0.5, crossing + 0.5, 201)
        step_coarse = np.abs(np.diff(coarse, axis=0)).max()
        # refine 10x: a smooth function's largest step shrinks ~10x; a pole
        # (or the old where-guard plateau jump) would keep an O(1) step.
        fine = _sweep(crossing - 0.5, crossing + 0.5, 2001)
        step_fine = np.abs(np.diff(fine, axis=0)).max()
        assert step_fine < 0.2 * step_coarse, (
            f"steps do not shrink under refinement (coarse {step_coarse:.4f} "
            f"-> fine {step_fine:.4f}): discontinuity or pole at the crossing"
        )

    def test_floor_does_not_disturb_dense_parity_regime(self) -> None:
        """In the in-design regime (all logits >= phantom_logit, count >= 0)
        the floor's correction is ~exp(-2*shift) relative -- invisible at
        the 1e-12 level used by the dense-parity suites.
        """
        rng = np.random.default_rng(3)
        data = rng.normal(size=7)  # normal-scale logits, shift 20 below
        seg = np.zeros(7, dtype=np.int64)
        w_floor = segment_softmax(
            data, seg, 1, phantom_count=np.array([3.0]), phantom_logit=-20.0
        )
        # reference: the exact fixed-width softmax with 3 phantom slots
        full = np.concatenate([data, np.full(3, -20.0)])
        ref = np.exp(full - full.max())
        ref = ref / ref.sum()
        np.testing.assert_allclose(w_floor, ref[:7], rtol=1e-12, atol=1e-15)
