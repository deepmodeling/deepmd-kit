import unittest

import numpy as np

from deepmd.utils.batch_size import AutoBatchSize
from deepmd.utils.errors import OutOfMemoryError

class TestAutoBatchSize(unittest.TestCase):
    def oom(self, batch_size, start_index):
        if batch_size >= 512:
            raise OutOfMemoryError
        return batch_size, np.zeros((batch_size, 2))

    def test_execute_oom(self):
        # initial batch size 256 = 128 * 2
        auto_batch_size = AutoBatchSize(256, 2.)
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
    
    def test_execute_all(self):
        dd1 = np.zeros((10000, 2, 1))
        auto_batch_size = AutoBatchSize(256, 2.)
        dd2 = auto_batch_size.execute_all(np.array, 10000, 2, dd1)
        np.testing.assert_equal(dd1, dd2)
