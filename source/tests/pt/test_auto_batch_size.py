# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)


class TestAutoBatchSize(unittest.TestCase):
    def test_execute_all(self) -> None:
        dd0 = np.zeros((10000, 2, 1, 3, 4))
        dd1 = np.ones((10000, 2, 1, 3, 4))
        auto_batch_size = AutoBatchSize(256, 2.0)

        def func(dd1):
            return np.zeros_like(dd1), np.ones_like(dd1)

        dd2 = auto_batch_size.execute_all(func, 10000, 2, dd1)
        np.testing.assert_equal(dd0, dd2[0])
        np.testing.assert_equal(dd1, dd2[1])

    def test_execute_all_dict(self) -> None:
        dd0 = np.zeros((10000, 2, 1, 3, 4))
        dd1 = np.ones((10000, 2, 1, 3, 4))
        auto_batch_size = AutoBatchSize(256, 2.0)

        def func(dd1):
            return {
                "foo": np.zeros_like(dd1),
                "bar": np.ones_like(dd1),
            }

        dd2 = auto_batch_size.execute_all(func, 10000, 2, dd1)
        np.testing.assert_equal(dd0, dd2["foo"])
        np.testing.assert_equal(dd1, dd2["bar"])
