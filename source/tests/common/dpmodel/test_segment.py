# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.segment import (
    segment_mean,
    segment_sum,
)


class TestSegment(unittest.TestCase):
    def test_segment_sum_1d_values(self) -> None:
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        seg = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        out = segment_sum(data, seg, 3)
        np.testing.assert_allclose(out, np.array([[3.0], [7.0], [5.0]]))

    def test_segment_sum_matrix_values(self) -> None:
        # (E, 3, 3) per-edge tensors aggregate per segment
        data = np.ones((4, 3, 3))
        seg = np.array([0, 0, 0, 1], dtype=np.int64)
        out = segment_sum(data, seg, 2)
        self.assertEqual(out.shape, (2, 3, 3))
        np.testing.assert_allclose(out[0], 3.0 * np.ones((3, 3)))
        np.testing.assert_allclose(out[1], np.ones((3, 3)))

    def test_segment_sum_empty_segment_is_zero(self) -> None:
        data = np.array([[1.0], [2.0]])
        seg = np.array([0, 2], dtype=np.int64)  # segment 1 gets nothing
        out = segment_sum(data, seg, 3)
        np.testing.assert_allclose(out, np.array([[1.0], [0.0], [2.0]]))

    def test_segment_mean(self) -> None:
        data = np.array([[2.0], [4.0], [9.0]])
        seg = np.array([0, 0, 1], dtype=np.int64)
        out = segment_mean(data, seg, 2)
        np.testing.assert_allclose(out, np.array([[3.0], [9.0]]))

    def test_segment_mean_empty_segment_no_nan(self) -> None:
        data = np.array([[2.0], [4.0]])
        seg = np.array([0, 0], dtype=np.int64)
        out = segment_mean(data, seg, 2)  # segment 1 empty -> 0, not nan
        np.testing.assert_allclose(out, np.array([[3.0], [0.0]]))
