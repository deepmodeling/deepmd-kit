# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.pd.utils.auto_batch_size import (
    AutoBatchSize,
)


class TestAutoBatchSize(unittest.TestCase):
    def test_execute_all(self):
        dd0 = paddle.zeros((10000, 2, 1, 3, 4))
        dd1 = paddle.ones((10000, 2, 1, 3, 4))
        auto_batch_size = AutoBatchSize(256, 2.0)

        def func(dd1):
            return paddle.zeros_like(dd1), paddle.ones_like(dd1)

        dd2 = auto_batch_size.execute_all(func, 10000, 2, dd1)
        np.testing.assert_equal(dd0.numpy(), dd2[0].numpy())
        np.testing.assert_equal(dd1.numpy(), dd2[1].numpy())

    def test_execute_all_dict(self):
        dd0 = paddle.zeros((10000, 2, 1, 3, 4))
        dd1 = paddle.ones((10000, 2, 1, 3, 4))
        auto_batch_size = AutoBatchSize(256, 2.0)

        def func(dd1):
            return {
                "foo": paddle.zeros_like(dd1),
                "bar": paddle.ones_like(dd1),
            }

        dd2 = auto_batch_size.execute_all(func, 10000, 2, dd1)
        np.testing.assert_equal(dd0.numpy(), dd2["foo"].numpy())
        np.testing.assert_equal(dd1.numpy(), dd2["bar"].numpy())
