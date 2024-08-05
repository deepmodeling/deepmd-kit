# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import sys
import unittest

from deepmd.utils.batch_size import (
    AutoBatchSize,
)
from deepmd.utils.errors import (
    OutOfMemoryError,
)

if sys.version_info >= (3, 9):
    import array_api_strict as xp
else:
    raise unittest.SkipTest("array_api_strict doesn't support Python<=3.8")


class CustomizedAutoBatchSizeCPU(AutoBatchSize):
    def is_gpu_available(self):
        return False

    def is_oom_error(self, e):
        return isinstance(e, OutOfMemoryError)


class CustomizedAutoBatchSizeGPU(AutoBatchSize):
    def is_gpu_available(self):
        return True

    def is_oom_error(self, e):
        return isinstance(e, OutOfMemoryError)


class TestAutoBatchSize(unittest.TestCase):
    def oom(self, batch_size, start_index):
        if batch_size >= 512:
            raise OutOfMemoryError
        return batch_size, xp.zeros((batch_size, 2))

    def test_execute_oom_gpu(self):
        # initial batch size 256 = 128 * 2
        auto_batch_size = CustomizedAutoBatchSizeGPU(256, 2.0)
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

    def test_execute_oom_cpu(self):
        # initial batch size 256 = 128 * 2, nb is always 128
        auto_batch_size = CustomizedAutoBatchSizeCPU(256, 2.0)
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
    def test_execute_oom_environment_variables(self):
        # DP_INFER_BATCH_SIZE = 256 = 128 * 2, nb is always 128
        auto_batch_size = CustomizedAutoBatchSizeGPU(999, 2.0)
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

    def test_execute_all(self):
        dd1 = xp.zeros((10000, 2, 1))
        auto_batch_size = CustomizedAutoBatchSizeGPU(256, 2.0)
        dd2 = auto_batch_size.execute_all(xp.asarray, 10000, 2, dd1)
        assert xp.all(dd1 == dd2)

    def test_execute_all_dict(self):
        dd0 = xp.zeros((10000, 2, 1, 3, 4))
        dd1 = xp.ones((10000, 2, 1, 3, 4))
        auto_batch_size = CustomizedAutoBatchSizeGPU(256, 2.0)

        def func(dd1):
            return {
                "foo": xp.zeros_like(dd1),
                "bar": xp.ones_like(dd1),
            }

        dd2 = auto_batch_size.execute_all(func, 10000, 2, dd1)
        assert xp.all(dd0 == dd2["foo"])
        assert xp.all(dd1 == dd2["bar"])

    def test_execute_all_dict_oom(self):
        # to reproduce #4036 when commenting "if n_batch == 0: continue"
        dd0 = xp.zeros((10, 2, 1, 3, 4))
        dd1 = xp.ones((10, 2, 1, 3, 4))
        auto_batch_size = CustomizedAutoBatchSizeGPU(4, 2.0)

        def func(dd1):
            if dd1.shape[0] >= 2:
                raise OutOfMemoryError
            return {
                "foo": xp.zeros_like(dd1),
                "bar": xp.ones_like(dd1),
            }

        dd2 = auto_batch_size.execute_all(func, 10, 2, dd1)
        assert xp.all(dd0 == dd2["foo"])
        assert xp.all(dd1 == dd2["bar"])
