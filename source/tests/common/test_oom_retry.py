# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.utils.batch_size import (
    AutoBatchSize,
    RetrySignal,
)
from deepmd.utils.errors import (
    OutOfMemoryError,
)


class CustomizedAutoBatchSizeGPU(AutoBatchSize):
    def is_gpu_available(self) -> bool:
        return True

    def is_oom_error(self, e):
        return isinstance(e, OutOfMemoryError)


class TestOOMRetry(unittest.TestCase):
    def test_execute_oom_retry_mode_raises_retry_signal(self) -> None:
        auto_batch_size = CustomizedAutoBatchSizeGPU(256, 2.0)

        oom = OutOfMemoryError("oom")

        def executor(batch_size: int, start_index: int) -> tuple[int, None]:
            raise oom

        auto_batch_size.set_oom_retry_mode(True)
        with self.assertRaises(RetrySignal) as context:
            auto_batch_size.execute(executor, 0, 1)
        self.assertIs(context.exception.__cause__, oom)
        self.assertEqual(auto_batch_size.current_batch_size, 128)

    def test_execute_oom_retry_mode_false_returns_zero(self) -> None:
        auto_batch_size = CustomizedAutoBatchSizeGPU(256, 2.0)

        def executor(batch_size: int, start_index: int) -> tuple[int, None]:
            raise OutOfMemoryError("oom")

        auto_batch_size.set_oom_retry_mode(False)
        n_batch, result = auto_batch_size.execute(executor, 0, 1)
        self.assertEqual(n_batch, 0)
        self.assertIsNone(result)
        self.assertEqual(auto_batch_size.current_batch_size, 128)


if __name__ == "__main__":
    unittest.main()
