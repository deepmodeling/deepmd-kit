# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

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


class DummyAutoBatchSize:
    def __init__(self) -> None:
        self.oom_retry_mode = False
        self.modes: list[bool] = []

    def set_oom_retry_mode(self, enable: bool) -> None:
        self.oom_retry_mode = enable
        self.modes.append(enable)


class DummyModel:
    def __init__(self) -> None:
        self.descriptor_hook_calls: list[bool] = []
        self.fitting_hook_calls: list[bool] = []

    def set_eval_descriptor_hook(self, enable: bool) -> None:
        self.descriptor_hook_calls.append(enable)

    def set_eval_fitting_last_layer_hook(self, enable: bool) -> None:
        self.fitting_hook_calls.append(enable)

    def eval_descriptor(self) -> list[int]:
        return [1, 2, 3]

    def eval_fitting_last_layer(self) -> list[int]:
        return [4, 5, 6]


class DummyDeepEval:
    def __init__(self, fail_once: bool = False, runtime_error: bool = False) -> None:
        self.auto_batch_size = DummyAutoBatchSize()
        self.model = DummyModel()
        self.dp = {"model": {"Default": self.model}}
        self.fail_once = fail_once
        self.runtime_error = runtime_error
        self.eval_calls = 0

    def eval(self, *args: Any, **kwargs: Any) -> None:
        self.eval_calls += 1
        if self.runtime_error:
            raise RuntimeError("non-retry failure")
        if self.fail_once and self.eval_calls == 1:
            raise RetrySignal

    def eval_descriptor(self, *args: Any, **kwargs: Any) -> list[int]:
        model = self.dp["model"]["Default"]
        if self.auto_batch_size is not None:
            self.auto_batch_size.set_oom_retry_mode(True)
        model.set_eval_descriptor_hook(True)
        retry = False
        try:
            self.eval(*args, **kwargs)
            descriptor = model.eval_descriptor()
        except RetrySignal:
            retry = True
        finally:
            model.set_eval_descriptor_hook(False)
            if self.auto_batch_size is not None:
                self.auto_batch_size.set_oom_retry_mode(False)
        if retry:
            return self.eval_descriptor(*args, **kwargs)
        return descriptor

    def eval_fitting_last_layer(self, *args: Any, **kwargs: Any) -> list[int]:
        model = self.dp["model"]["Default"]
        if self.auto_batch_size is not None:
            self.auto_batch_size.set_oom_retry_mode(True)
        model.set_eval_fitting_last_layer_hook(True)
        retry = False
        try:
            self.eval(*args, **kwargs)
            fitting = model.eval_fitting_last_layer()
        except RetrySignal:
            retry = True
        finally:
            model.set_eval_fitting_last_layer_hook(False)
            if self.auto_batch_size is not None:
                self.auto_batch_size.set_oom_retry_mode(False)
        if retry:
            return self.eval_fitting_last_layer(*args, **kwargs)
        return fitting


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

    def test_eval_descriptor_retry_clears_hook_between_attempts(self) -> None:
        deep_eval = DummyDeepEval(fail_once=True)
        self.assertEqual(deep_eval.eval_descriptor(), [1, 2, 3])
        self.assertEqual(deep_eval.eval_calls, 2)
        self.assertEqual(
            deep_eval.model.descriptor_hook_calls,
            [True, False, True, False],
        )
        self.assertFalse(deep_eval.auto_batch_size.oom_retry_mode)

    def test_eval_fitting_last_layer_retry_clears_hook_between_attempts(self) -> None:
        deep_eval = DummyDeepEval(fail_once=True)
        self.assertEqual(deep_eval.eval_fitting_last_layer(), [4, 5, 6])
        self.assertEqual(deep_eval.eval_calls, 2)
        self.assertEqual(
            deep_eval.model.fitting_hook_calls,
            [True, False, True, False],
        )
        self.assertFalse(deep_eval.auto_batch_size.oom_retry_mode)

    def test_eval_descriptor_runtime_error_clears_state(self) -> None:
        deep_eval = DummyDeepEval(runtime_error=True)
        with self.assertRaisesRegex(RuntimeError, "non-retry failure"):
            deep_eval.eval_descriptor()
        self.assertEqual(deep_eval.model.descriptor_hook_calls, [True, False])
        self.assertFalse(deep_eval.auto_batch_size.oom_retry_mode)

    def test_eval_fitting_last_layer_runtime_error_clears_state(self) -> None:
        deep_eval = DummyDeepEval(runtime_error=True)
        with self.assertRaisesRegex(RuntimeError, "non-retry failure"):
            deep_eval.eval_fitting_last_layer()
        self.assertEqual(deep_eval.model.fitting_hook_calls, [True, False])
        self.assertFalse(deep_eval.auto_batch_size.oom_retry_mode)


if __name__ == "__main__":
    unittest.main()
