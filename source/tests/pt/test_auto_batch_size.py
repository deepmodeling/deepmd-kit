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


class TestCudaAdaptiveBatching(unittest.TestCase):
    def setUp(self) -> None:
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, {"DP_INFER_BATCH_SIZE": "0"}))
        stack.enter_context(mock.patch("torch.cuda.is_available", return_value=True))
        stack.enter_context(
            mock.patch("deepmd.pt.utils.env.DEVICE", torch.device("cuda:0"))
        )
        stack.enter_context(mock.patch("torch.cuda.device", return_value=nullcontext()))
        stack.enter_context(mock.patch("torch.cuda.current_device", return_value=0))
        stack.enter_context(
            mock.patch("torch.cuda.get_allocator_backend", return_value="native")
        )
        # A successful hot path must not inspect or mutate allocator state.
        for name in (
            "memory_stats_as_nested_dict",
            "memory_snapshot",
            "mem_get_info",
            "reset_peak_memory_stats",
            "set_per_process_memory_fraction",
        ):
            stack.enter_context(
                mock.patch.object(
                    torch.cuda, name, side_effect=AssertionError(f"unexpected {name}")
                )
            )
        self.empty_cache = stack.enter_context(mock.patch("torch.cuda.empty_cache"))

    def test_short_systems_reuse_atom_budget_without_calibration(self) -> None:
        auto = AutoBatchSize(1024)
        data = np.arange(16).reshape(8, 2)
        attempts = []

        def evaluate(batch: np.ndarray) -> np.ndarray:
            attempts.append(len(batch))
            return batch.copy()

        for natoms in (100, 101, 100, 101):
            result = auto.execute_all(evaluate, len(data), natoms, data)
            np.testing.assert_array_equal(result, data)
        self.assertEqual(attempts, [8, 8, 8, 8])
        self.assertEqual(auto.current_batch_size, 1024)
        self.empty_cache.assert_not_called()

    def test_fixed_workspace_does_not_trap_batch_at_one_frame(self) -> None:
        auto = AutoBatchSize(100)
        data = np.arange(128).reshape(64, 2)
        attempts = []

        def evaluate(batch: np.ndarray) -> np.ndarray:
            attempts.append(len(batch))
            # Six frames fit; one-frame extrapolation would admit only one.
            if 600 + 50 * len(batch) > 900:
                raise torch.cuda.OutOfMemoryError()
            return batch.copy()

        for _ in range(2):
            result = auto.execute_all(evaluate, len(data), 100, data)
            np.testing.assert_array_equal(result, data)
        self.assertEqual(attempts[:5], [1, 2, 4, 8, 4])
        self.assertIn(auto.current_batch_size, (400, 800))
        self.assertEqual(auto._reprobe_interval, 8)
        self.assertGreater(attempts.count(4), attempts.count(8))
        self.assertEqual(self.empty_cache.call_count, attempts.count(8))

    def test_changed_memory_pressure_backs_off_and_reuses_the_bound(self) -> None:
        auto = AutoBatchSize(8)
        data = np.arange(128).reshape(64, 2)
        capacity = 8
        attempts = []

        def evaluate(batch: np.ndarray) -> np.ndarray:
            attempts.append(len(batch))
            if len(batch) > capacity:
                raise torch.cuda.OutOfMemoryError()
            return batch.copy()

        result = auto.execute_all(evaluate, len(data), 2, data)
        np.testing.assert_array_equal(result, data)
        self.assertIn(auto.current_batch_size, (16, 32))
        capacity = 2
        attempts.clear()
        for _ in range(2):
            result = auto.execute_all(evaluate, len(data), 2, data)
            np.testing.assert_array_equal(result, data)
        first_success = attempts.index(2)
        self.assertGreaterEqual(attempts[0], 8)
        self.assertEqual(attempts[first_success - 1], 4)
        self.assertLessEqual(max(attempts[first_success:]), 4)
        self.assertGreater(attempts.count(2), attempts.count(4))
        self.assertEqual(auto._reprobe_interval, 8)

    def test_dense_to_sparse_recovers_even_at_maximum_probe_interval(self) -> None:
        auto = AutoBatchSize(2)
        capacity = 2
        attempts = []

        def evaluate(nframes: int, start: int) -> tuple[int, None]:
            attempts.append((nframes, start))
            if nframes > capacity:
                raise torch.cuda.OutOfMemoryError()
            return nframes, None

        # A stationary dense workload damps repeated probes, but not forever.
        for _ in range(80):
            auto.execute(evaluate, 0, 1)
        self.assertEqual(auto._reprobe_interval, 8)
        failures = [i for i, (n, _) in enumerate(attempts) if n > capacity]
        self.assertTrue(
            all(b - a == 9 for a, b in zip(failures[-4:-1], failures[-3:], strict=True))
        )
        capacity = 16  # Same atom count, less memory per frame.
        recovered = False
        for _ in range(11):  # At most eight successes, then 4 -> 8 -> 16.
            n, _ = auto.execute(evaluate, 0, 1)
            if n == capacity:
                recovered = True
                break
        self.assertTrue(recovered)
        self.assertEqual(auto._reprobe_interval, 8)

    def test_initial_backoff_and_clipped_probe_preserve_the_failed_bound(self) -> None:
        auto = AutoBatchSize(64)
        capacity = 2
        attempts = []

        def evaluate(nframes: int, start: int) -> tuple[int, None]:
            attempts.append((nframes, start))
            if nframes > capacity:
                raise torch.cuda.OutOfMemoryError()
            return nframes, None

        for _ in range(5):
            self.assertEqual(auto.execute(evaluate, 17, 1), (0, None))
        self.assertEqual(attempts, [(n, 17) for n in (64, 32, 16, 8, 4)])
        self.assertEqual(auto._reprobe_interval, 2)
        auto.execute(evaluate, 17, 1)
        auto.execute(evaluate, 19, 1)
        self.assertEqual(auto.current_batch_size, 4)
        auto.execute(evaluate, 21, 1, max_nframes=1)
        self.assertEqual(auto.minimal_not_working_batch_size, 4)
        self.assertEqual(auto.current_batch_size, 4)
        capacity = 8
        auto.execute(evaluate, 22, 1)
        self.assertEqual(auto.current_batch_size, 8)
        self.assertGreater(auto.minimal_not_working_batch_size, 8)

    def test_short_tails_and_atom_count_changes_do_not_age_a_probe(self) -> None:
        auto = AutoBatchSize(400)
        auto.execute(mock.Mock(side_effect=torch.cuda.OutOfMemoryError()), 0, 100)
        self.assertEqual(auto.current_batch_size, 200)
        for natoms in (50, 51) * 8:
            auto.execute(lambda n, start: (n, None), 0, natoms, max_nframes=1)
        self.assertEqual(auto.current_batch_size, 200)
        self.assertEqual(auto.minimal_not_working_batch_size, 400)
        self.assertEqual(auto._successful_batches, 0)
        self.assertEqual(auto._reprobe_interval, 2)

    def test_alternating_capacity_does_not_reset_probe_interval(self) -> None:
        auto = AutoBatchSize(2)
        capacity = 2

        def evaluate(nframes: int, start: int) -> tuple[int, None]:
            if nframes > capacity:
                raise torch.cuda.OutOfMemoryError()
            return nframes, None

        auto.execute(evaluate, 0, 1)  # 2 -> 4.
        auto.execute(evaluate, 0, 1)  # First dense OOM.
        for interval, next_interval in ((2, 4), (4, 8), (8, 8)):
            for _ in range(interval):
                auto.execute(evaluate, 0, 1)
            self.assertEqual(auto.current_batch_size, 4)
            self.assertEqual(auto._reprobe_interval, next_interval)
            capacity = 4
            auto.execute(evaluate, 0, 1)  # Sparse probe succeeds and grows to 8.
            self.assertEqual(auto.current_batch_size, 8)
            capacity = 2
            auto.execute(evaluate, 0, 1)
            auto.execute(evaluate, 0, 1)  # Dense backoff: 8 -> 4 -> 2.
            self.assertEqual(auto.current_batch_size, 2)
            self.assertEqual(auto._reprobe_interval, next_interval)

    def test_explicit_limit_keeps_oom_recovery_and_retry_mode(self) -> None:
        with mock.patch.dict(os.environ, {"DP_INFER_BATCH_SIZE": "8"}):
            auto = AutoBatchSize()
        auto.set_oom_retry_mode(True)
        data = np.arange(16).reshape(8, 2)
        attempts = []

        def evaluate(batch: np.ndarray) -> np.ndarray:
            attempts.append(len(batch))
            if len(batch) > 2:
                raise torch.cuda.OutOfMemoryError()
            return batch.copy()

        with self.assertRaises(RetrySignal):
            auto.execute_all(evaluate, len(data), 2, data)
        self.assertEqual(auto.current_batch_size, 4)
        result = auto.execute_all(evaluate, len(data), 2, data)
        np.testing.assert_array_equal(result, data)
        self.assertEqual(attempts, [4, 2, 2, 2, 2])
        self.empty_cache.assert_called_once()

    def test_single_frame_oom_and_unrelated_errors_propagate(self) -> None:
        data = np.ones((1, 2))
        with self.assertRaises(OutOfMemoryError):
            AutoBatchSize().execute_all(
                mock.Mock(side_effect=torch.cuda.OutOfMemoryError()), 1, 2, data
            )
        failure = RuntimeError("invalid model input")
        with self.assertRaisesRegex(RuntimeError, "invalid model input"):
            AutoBatchSize().execute_all(mock.Mock(side_effect=failure), 1, 2, data)
        self.empty_cache.assert_called_once()


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
