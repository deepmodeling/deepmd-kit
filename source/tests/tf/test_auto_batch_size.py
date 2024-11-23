# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np

from deepmd.tf.utils.batch_size import (
    AutoBatchSize,
)
from deepmd.tf.utils.errors import (
    OutOfMemoryError,
)


class TestAutoBatchSize(unittest.TestCase):
    def oom(self, batch_size, start_index):
        if batch_size >= 512:
            raise OutOfMemoryError
        return batch_size, np.zeros((batch_size, 2))

    @unittest.mock.patch("tensorflow.compat.v1.test.is_gpu_available")
    def test_execute_oom_gpu(self, mock_is_gpu_available) -> None:
        mock_is_gpu_available.return_value = True
        # initial batch size 256 = 128 * 2
        auto_batch_size = AutoBatchSize(256, 2.0)
        # no error - 128
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        # no error - 256
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 256)
        self.assertEqual(result.shape, (256, 2))
        # error - 512 return 0, None
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 0)
        self.assertIsNone(result)
        # 256 again
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 256)
        self.assertEqual(result.shape, (256, 2))
        # 256 again
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 256)
        self.assertEqual(result.shape, (256, 2))

    @unittest.mock.patch("tensorflow.compat.v1.test.is_gpu_available")
    @unittest.mock.patch("tensorflow.compat.v1.config.experimental.get_visible_devices")
    def test_execute_oom_cpu(
        self, mock_is_gpu_available, mock_get_visible_devices
    ) -> None:
        mock_is_gpu_available.return_value = False
        mock_get_visible_devices.return_value = []
        # initial batch size 256 = 128 * 2, nb is always 128
        auto_batch_size = AutoBatchSize(256, 2.0)
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))

    @unittest.mock.patch.dict(os.environ, {"DP_INFER_BATCH_SIZE": "256"}, clear=True)
    def test_execute_oom_environment_variables(self) -> None:
        # DP_INFER_BATCH_SIZE = 256 = 128 * 2, nb is always 128
        auto_batch_size = AutoBatchSize(999, 2.0)
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))
        nb, result = auto_batch_size.execute(self.oom, 1, 2)
        self.assertEqual(nb, 128)
        self.assertEqual(result.shape, (128, 2))

    def test_execute_all(self) -> None:
        dd1 = np.zeros((10000, 2, 1))
        auto_batch_size = AutoBatchSize(256, 2.0)
        dd2 = auto_batch_size.execute_all(np.array, 10000, 2, dd1)
        np.testing.assert_equal(dd1, dd2)
