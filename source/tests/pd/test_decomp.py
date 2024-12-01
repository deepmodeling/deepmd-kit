# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.pd.utils import (
    decomp,
)

from ..seed import (
    GLOBAL_SEED,
)


class TestDecomp(unittest.TestCase):
    def setUp(self):
        paddle.seed(GLOBAL_SEED)

    def test_scatter_reduce_decomp(self):
        raw_api = paddle.put_along_axis
        decomp_api = decomp.scatter_reduce
        raw_input = paddle.randn([100, 100], "float32")
        axis = 0
        raw_index = paddle.randint(0, 100, [100, 100], "int64")
        raw_values = paddle.randn([100, 100], "float32")
        raw_output = raw_api(raw_input, raw_index, raw_values, axis=axis, reduce="add")
        decomp_output = decomp_api(
            raw_input, axis, raw_index, src=raw_values, reduce="sum"
        )

        np.testing.assert_allclose(
            raw_output.numpy(),
            decomp_output.numpy(),
            2e-5,
            1e-7,
        )

    def test_sec(self):
        shape = [10, 3]
        length = shape[0]
        size = 3

        split_sections = decomp.sec(length, size)
        assert split_sections == [3, 3, 3, 1]

    def test_masked_add_(self):
        decomp_api = decomp.masked_add_

        raw_input = paddle.randn([10, 10], "float32")
        raw_mask = paddle.randint(0, 2, [10, 10]).astype("bool")
        add_values = paddle.randn([10, 10], "float32")
        raw_output = raw_input.clone()

        for i in range(raw_input.shape[0]):
            for j in range(raw_input.shape[1]):
                if raw_mask[i][j]:
                    raw_output[i][j] += add_values[i][j]

        decomp_output = decomp_api(raw_input, raw_mask, add_values[raw_mask])

        np.testing.assert_equal(
            raw_output.numpy(),
            decomp_output.numpy(),  # inplace
        )

        np.testing.assert_equal(
            raw_output.numpy(),
            raw_input.numpy(),  # inplace
        )
