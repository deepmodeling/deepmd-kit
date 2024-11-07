# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)


class TestAutoBatchSize(unittest.TestCase):
    def test_execute_all(self):
        dd0 = torch.zeros((10000, 2, 1, 3, 4))
        dd1 = torch.ones((10000, 2, 1, 3, 4))
        auto_batch_size = AutoBatchSize(256, 2.0)

        def func(dd1):
            return torch.zeros_like(dd1), torch.ones_like(dd1)

        dd2 = auto_batch_size.execute_all(func, 10000, 2, dd1)
        np.testing.assert_equal(dd0.cpu().numpy(), dd2[0].cpu().numpy())
        np.testing.assert_equal(dd1.cpu().numpy(), dd2[1].cpu().numpy())

    def test_execute_all_dict(self):
        dd0 = torch.zeros((10000, 2, 1, 3, 4))
        dd1 = torch.ones((10000, 2, 1, 3, 4))
        auto_batch_size = AutoBatchSize(256, 2.0)

        def func(dd1):
            return {
                "foo": torch.zeros_like(dd1),
                "bar": torch.ones_like(dd1),
            }

        dd2 = auto_batch_size.execute_all(func, 10000, 2, dd1)
        np.testing.assert_equal(dd0.cpu().numpy(), dd2["foo"].cpu().numpy())
        np.testing.assert_equal(dd1.cpu().numpy(), dd2["bar"].cpu().numpy())
