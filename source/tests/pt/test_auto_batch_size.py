# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest
from contextlib import (
    ExitStack,
    nullcontext,
)
from unittest import (
    mock,
)

import numpy as np
import torch

from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.utils.batch_size import (
    RetrySignal,
)
from deepmd.utils.errors import (
    OutOfMemoryError,
)


class _CudaAllocator:
    """Deterministic allocator with retained workspace and graph-private memory."""

    def __init__(self) -> None:
        self.total = 1000
        self.live = 100
        self.private = 0
        self.private_active = 0
        self.retained = self.live
        self.reserved = self.live
        self.peak = self.live
        self.allocated_total = self.live
        self.allocations = 0
        self.frees = 0
        self.ooms = 0
        self.frame_bytes = 100
        self.batches = []

    def stats(self, device: int) -> dict:
        return {
            "allocated_bytes": {
                "all": {
                    "current": self.live + self.private_active,
                    "peak": self.peak,
                    "allocated": self.allocated_total,
                }
            },
            "active_bytes": {"all": {"current": self.live + self.private_active}},
            "reserved_bytes": {"all": {"current": self.reserved + self.private}},
            "reserved_bytes_by_private_pools": {
                (0, 1): {"all": {"current": self.private}}
            },
            "num_device_alloc": self.allocations,
            "num_device_free": self.frees,
        }

    def mem_get_info(self, device: int) -> tuple[int, int]:
        return self.total - self.reserved - self.private, self.total

    def empty_cache(self) -> None:
        if self.reserved > self.retained:
            self.reserved = self.retained
            self.frees += 1

    def snapshot(self) -> list[dict]:
        return [
            {
                "device": 1,
                "segment_pool_id": (0, 0),
                "total_size": self.reserved,
                "active_size": self.live,
            },
            {
                "device": 1,
                "segment_pool_id": (0, 1),
                "total_size": self.private,
                "active_size": self.private_active,
            },
            {
                "device": 0,
                "segment_pool_id": (0, 0),
                "total_size": 10000,
                "active_size": 0,
            },
        ]

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        self.batches.append(len(data))
        required = self.live + len(data) * self.frame_bytes
        if required + self.private > self.total:
            self.ooms += 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        if required > self.reserved:
            self.reserved = required
            self.allocations += 1
        self.peak = max(self.peak, required + self.private_active)
        self.allocated_total += len(data) * self.frame_bytes
        return data.copy()


class TestCudaMemoryBatching(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = _CudaAllocator()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, {"DP_INFER_BATCH_SIZE": "0"}))
        stack.enter_context(
            mock.patch(
                "deepmd.pt.utils.auto_batch_size.env.DEVICE", torch.device("cuda:1")
            )
        )
        stack.enter_context(mock.patch("torch.cuda.is_available", return_value=True))
        stack.enter_context(
            mock.patch("torch.cuda.get_allocator_backend", return_value="native")
        )
        stack.enter_context(mock.patch("torch.cuda.current_device", return_value=1))
        self.device = stack.enter_context(
            mock.patch("torch.cuda.device", return_value=nullcontext())
        )
        self.stats = stack.enter_context(
            mock.patch(
                "torch.cuda.memory_stats_as_nested_dict",
                side_effect=self.allocator.stats,
            )
        )
        stack.enter_context(
            mock.patch(
                "torch.cuda.mem_get_info", side_effect=self.allocator.mem_get_info
            )
        )
        stack.enter_context(
            mock.patch("torch.cuda.get_per_process_memory_fraction", return_value=1.0)
        )
        self.empty_cache = stack.enter_context(
            mock.patch("torch.cuda.empty_cache", side_effect=self.allocator.empty_cache)
        )
        stack.enter_context(
            mock.patch(
                "torch.cuda.memory_snapshot", side_effect=self.allocator.snapshot
            )
        )
        self.reset_peak = stack.enter_context(
            mock.patch("torch.cuda.reset_peak_memory_stats")
        )
        self.set_fraction = stack.enter_context(
            mock.patch("torch.cuda.set_per_process_memory_fraction")
        )

    def _evaluate(self, auto: AutoBatchSize, natoms: int = 2) -> np.ndarray:
        data = np.arange(128).reshape(64, 2)
        result = auto.execute_all(self.allocator.evaluate, len(data), natoms, data)
        np.testing.assert_array_equal(result, data)
        return result

    def test_growth_stops_before_allocator_oom(self) -> None:
        """The same workload crosses OOM under unbounded doubling, but not admission."""
        legacy = AutoBatchSize(128)
        data = np.arange(128).reshape(64, 2)
        with mock.patch.object(legacy, "_get_batch_budget", return_value=None):
            legacy.execute_all(self.allocator.evaluate, len(data), 2, data)
        self.assertGreater(self.allocator.ooms, 0)
        self.allocator.empty_cache()
        self.allocator.batches.clear()
        self.allocator.ooms = 0
        self._evaluate(AutoBatchSize(128))
        self.assertEqual(self.allocator.ooms, 0)
        self.assertEqual(self.allocator.batches[0], 1)
        self.assertEqual(max(self.allocator.batches), 8)
        self.device.assert_called_with(torch.device("cuda:1"))
        self.reset_peak.assert_not_called()
        self.set_fraction.assert_not_called()

    def test_private_pool_is_not_available_workspace(self) -> None:
        self.allocator.private = 500
        for private_active in (0, 500):
            with self.subTest(private_active=private_active):
                self.allocator.private_active = private_active
                self.allocator.batches.clear()
                self._evaluate(AutoBatchSize(128))
                self.assertEqual(self.allocator.ooms, 0)
                self.assertEqual(max(self.allocator.batches), 3)

    def test_calibration_preserves_the_requested_atom_budget(self) -> None:
        data = np.arange(16).reshape(8, 2)
        result = AutoBatchSize(16).execute_all(
            self.allocator.evaluate, len(data), 2, data
        )
        np.testing.assert_array_equal(result, data)
        self.assertEqual(self.allocator.batches, [1, 7])
        self.assertEqual(self.allocator.ooms, 0)

    def test_calibration_survives_a_stale_global_peak(self) -> None:
        self.allocator.peak = self.allocator.total
        self._evaluate(AutoBatchSize(128))
        self.assertEqual(self.allocator.ooms, 0)
        self.assertEqual(max(self.allocator.batches), 8)
        self.assertEqual(self.allocator.peak, self.allocator.total)

    def test_calibration_accounts_for_unreleasable_cache(self) -> None:
        self.allocator.reserved = self.allocator.retained = 700
        self.allocator.peak = self.allocator.total
        self._evaluate(AutoBatchSize(128))
        self.assertEqual(self.allocator.ooms, 0)
        self.assertEqual(max(self.allocator.batches), 8)
        self.assertEqual(self.allocator.peak, self.allocator.total)

    def test_reuses_calibration_and_remeasures_changed_workload(self) -> None:
        auto = AutoBatchSize(128)
        self._evaluate(auto)
        cache_releases = self.empty_cache.call_count
        self.allocator.batches.clear()
        self._evaluate(auto)
        self.assertEqual(self.allocator.batches[0], 8)
        self.assertEqual(self.empty_cache.call_count, cache_releases)
        self.allocator.frame_bytes = 150
        self.allocator.batches.clear()
        self._evaluate(auto, natoms=4)
        self.assertEqual(self.allocator.batches[0], 1)
        self.assertEqual(max(self.allocator.batches), 5)
        self.assertEqual(self.allocator.ooms, 0)

    def test_recalibrates_after_training_claims_device_memory(self) -> None:
        auto = AutoBatchSize(128)
        self._evaluate(auto)
        self.allocator.empty_cache()
        self.allocator.private = 700
        self.allocator.allocations += 1
        self.allocator.batches.clear()
        self._evaluate(auto)
        self.assertEqual(max(self.allocator.batches), 1)
        self.assertEqual(self.allocator.ooms, 0)

    def test_fixed_budget_and_single_frames_do_not_profile(self) -> None:
        with mock.patch.dict(os.environ, {"DP_INFER_BATCH_SIZE": "8"}):
            self._evaluate(AutoBatchSize())
        self.assertEqual(self.allocator.batches[0], 4)
        self.stats.assert_not_called()
        auto = AutoBatchSize()
        data = np.ones((1, 2))
        auto.execute_all(self.allocator.evaluate, 1, 2, data)
        self.stats.assert_not_called()

    def test_single_frame_oom_and_unrelated_errors_propagate(self) -> None:
        self.allocator.frame_bytes = self.allocator.total
        with self.assertRaises(OutOfMemoryError):
            self._evaluate(AutoBatchSize())
        failure = RuntimeError("invalid model input")
        with self.assertRaisesRegex(RuntimeError, "invalid model input"):
            AutoBatchSize().execute_all(
                mock.Mock(side_effect=failure), 2, 2, np.ones((2, 2))
            )

    def test_unexpected_workspace_increase_keeps_oom_backoff(self) -> None:
        auto = AutoBatchSize(128)
        data = np.arange(128).reshape(64, 2)

        def evaluate(batch: np.ndarray) -> np.ndarray:
            if batch[0, 0] >= 2:
                self.allocator.frame_bytes = 200
            return self.allocator.evaluate(batch)

        result = auto.execute_all(evaluate, len(data), 2, data)
        np.testing.assert_array_equal(result, data)
        self.assertEqual(self.allocator.ooms, 1)

    def test_respects_existing_allocator_memory_fraction(self) -> None:
        with mock.patch("torch.cuda.get_per_process_memory_fraction", return_value=0.5):
            self._evaluate(AutoBatchSize(128))
        self.assertEqual(max(self.allocator.batches), 3)
        self.set_fraction.assert_not_called()

    def test_retry_signal_preserves_backoff(self) -> None:
        auto = AutoBatchSize(8)
        auto.set_oom_retry_mode(True)
        data = np.arange(128).reshape(64, 2)

        def evaluate(batch: np.ndarray) -> np.ndarray:
            if len(batch) > 2:
                self.allocator.frame_bytes = 350
            return self.allocator.evaluate(batch)

        with self.assertRaises(RetrySignal):
            auto.execute_all(evaluate, len(data), 2, data)
        self.assertEqual(auto.current_batch_size, 4)
        result = auto.execute_all(self.allocator.evaluate, len(data), 2, data)
        np.testing.assert_array_equal(result, data)
        self.assertEqual(self.allocator.ooms, 1)

    def test_cpu_and_async_allocator_keep_the_existing_policy(self) -> None:
        data = np.ones((8, 2))
        for device, backend in (("cpu", "native"), ("cuda:1", "cudaMallocAsync")):
            with (
                self.subTest(device=device, backend=backend),
                mock.patch(
                    "deepmd.pt.utils.auto_batch_size.env.DEVICE", torch.device(device)
                ),
                mock.patch("torch.cuda.get_allocator_backend", return_value=backend),
            ):
                result = AutoBatchSize().execute_all(np.copy, len(data), 2, data)
                np.testing.assert_array_equal(result, data)
        self.stats.assert_not_called()


class TestAutoBatchSize(unittest.TestCase):
    @mock.patch("deepmd.pt.utils.auto_batch_size.torch.cuda.empty_cache")
    def test_is_oom_error_cuda_message(self, empty_cache) -> None:
        auto_batch_size = AutoBatchSize(256, 2.0)

        self.assertTrue(
            auto_batch_size.is_oom_error(RuntimeError("CUDA out of memory."))
        )
        empty_cache.assert_called_once()

    @mock.patch("deepmd.pt.utils.auto_batch_size.torch.cuda.empty_cache")
    def test_is_oom_error_cublas_alloc_failed(self, empty_cache) -> None:
        auto_batch_size = AutoBatchSize(256, 2.0)

        self.assertTrue(
            auto_batch_size.is_oom_error(
                RuntimeError(
                    "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling "
                    "`cublasCreate(handle)`"
                )
            )
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
