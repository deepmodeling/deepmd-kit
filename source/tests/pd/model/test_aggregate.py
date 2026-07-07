# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test that ``aggregate`` preserves the input dtype (float64 must not downcast)."""

import unittest

import numpy as np
import paddle

from deepmd.pd.model.network.utils import (
    aggregate,
)


class TestAggregateDtype(unittest.TestCase):
    def _data(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        # rows 0,1 -> owner 0 ; row 2 -> owner 1
        data = paddle.to_tensor([[1.0], [2.0], [3.0]], dtype="float64")
        owners = paddle.to_tensor([0, 0, 1], dtype="int64")
        return data, owners

    def test_sum_preserves_float64(self) -> None:
        # exercises the shared ``output = paddle.zeros(..., dtype=data.dtype)``
        # allocation via the summation path
        data, owners = self._data()
        out = aggregate(data, owners, average=False, num_owner=2)
        self.assertEqual(out.dtype, paddle.float64)
        np.testing.assert_allclose(out.numpy().ravel(), [3.0, 3.0])
