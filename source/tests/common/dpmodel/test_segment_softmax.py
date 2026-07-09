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
