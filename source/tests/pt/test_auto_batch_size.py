# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest import (
    mock,
)

import numpy as np

from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)


class TestAutoBatchSize(unittest.TestCase):
    @mock.patch("deepmd.pt.utils.auto_batch_size.torch.cuda.empty_cache")
    def test_is_oom_error_cuda_message(self, empty_cache) -> None:
        auto_batch_size = AutoBatchSize(256, 2.0)

        self.assertTrue(
            auto_batch_size.is_oom_error(RuntimeError("CUDA out of memory."))
        )
        empty_cache.assert_called_once()

    @mock.patch("deepmd.pt.utils.auto_batch_size.torch.cuda.empty_cache")
    def test_is_oom_error_empty_runtime_error_from_cuda_oom(self, empty_cache) -> None:
        auto_batch_size = AutoBatchSize(256, 2.0)
        cause = RuntimeError("CUDA driver error: out of memory")
        error = RuntimeError()
        error.__cause__ = cause

        self.assertTrue(auto_batch_size.is_oom_error(error))
        empty_cache.assert_called_once()

    @mock.patch("deepmd.pt.utils.auto_batch_size.torch.cuda.empty_cache")
    def test_is_oom_error_aoti_wrapper(self, empty_cache) -> None:
        auto_batch_size = AutoBatchSize(256, 2.0)
        error = RuntimeError(
            "run_func_(...) API call failed at "
            "/tmp/torchinductor/model_container_runner.cpp"
        )

        self.assertTrue(auto_batch_size.is_oom_error(error))
        empty_cache.assert_called_once()

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
