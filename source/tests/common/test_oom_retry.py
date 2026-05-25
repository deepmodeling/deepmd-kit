# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from types import SimpleNamespace
from typing import (
    Any,
)
from unittest.mock import (
    MagicMock,
    call,
    patch,
)

import numpy as np

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

    def _make_backend(self, backend: str, method_name: str) -> tuple[Any, MagicMock]:
        try:
            if backend == "pt":
                from deepmd.pt.infer.deep_eval import DeepEval
            else:
                from deepmd.pd.infer.deep_eval import DeepEval
        except ModuleNotFoundError as exc:
            self.skipTest(f"{backend} backend dependencies are unavailable: {exc}")

        abstract_methods = getattr(DeepEval, "__abstractmethods__", frozenset())
        try:
            DeepEval.__abstractmethods__ = frozenset()
            deep_eval = object.__new__(DeepEval)
        finally:
            DeepEval.__abstractmethods__ = abstract_methods

        model = MagicMock()
        model.eval_descriptor.return_value = np.array([1, 2, 3])
        model.eval_fitting_last_layer.return_value = np.array([4, 5, 6])

        if backend == "pd" and method_name == "eval_descriptor":
            # Paddle eval_descriptor accepts either a ModelWrapper or a direct model.
            deep_eval.dp = model
        else:
            deep_eval.dp = SimpleNamespace(model={"Default": model})
        deep_eval.auto_batch_size = DummyAutoBatchSize()
        return deep_eval, model

    def _assert_retry_clears_hook_between_attempts(
        self,
        backend: str,
        method_name: str,
        hook_name: str,
        expected: np.ndarray,
    ) -> None:
        deep_eval, model = self._make_backend(backend, method_name)
        with patch.object(
            deep_eval, "eval", side_effect=[RetrySignal, None]
        ) as eval_mock:
            result = getattr(deep_eval, method_name)(
                coords=np.zeros((3, 1, 3)),
                cells=None,
                atom_types=np.array([0]),
            )
        self.assertEqual(eval_mock.call_count, 2)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(
            getattr(model, hook_name).call_args_list,
            [call(True), call(False), call(True), call(False)],
        )
        self.assertFalse(deep_eval.auto_batch_size.oom_retry_mode)
        self.assertEqual(deep_eval.auto_batch_size.modes, [True, False, True, False])

    def _assert_runtime_error_clears_state(
        self,
        backend: str,
        method_name: str,
        hook_name: str,
    ) -> None:
        deep_eval, model = self._make_backend(backend, method_name)
        with patch.object(
            deep_eval,
            "eval",
            side_effect=RuntimeError("non-retry failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "non-retry failure"):
                getattr(deep_eval, method_name)(
                    coords=np.zeros((3, 1, 3)),
                    cells=None,
                    atom_types=np.array([0]),
                )
        self.assertEqual(
            getattr(model, hook_name).call_args_list, [call(True), call(False)]
        )
        self.assertFalse(deep_eval.auto_batch_size.oom_retry_mode)
        self.assertEqual(deep_eval.auto_batch_size.modes, [True, False])

    def test_pt_eval_descriptor_retry_clears_hook_between_attempts(self) -> None:
        self._assert_retry_clears_hook_between_attempts(
            "pt",
            "eval_descriptor",
            "set_eval_descriptor_hook",
            np.array([1, 2, 3]),
        )

    def test_pt_eval_fitting_last_layer_retry_clears_hook_between_attempts(
        self,
    ) -> None:
        self._assert_retry_clears_hook_between_attempts(
            "pt",
            "eval_fitting_last_layer",
            "set_eval_fitting_last_layer_hook",
            np.array([4, 5, 6]),
        )

    def test_pd_eval_descriptor_retry_clears_hook_between_attempts(self) -> None:
        self._assert_retry_clears_hook_between_attempts(
            "pd",
            "eval_descriptor",
            "set_eval_descriptor_hook",
            np.array([1, 2, 3]),
        )

    def test_pd_eval_fitting_last_layer_retry_clears_hook_between_attempts(
        self,
    ) -> None:
        self._assert_retry_clears_hook_between_attempts(
            "pd",
            "eval_fitting_last_layer",
            "set_eval_fitting_last_layer_hook",
            np.array([4, 5, 6]),
        )

    def test_pt_eval_descriptor_runtime_error_clears_state(self) -> None:
        self._assert_runtime_error_clears_state(
            "pt",
            "eval_descriptor",
            "set_eval_descriptor_hook",
        )

    def test_pt_eval_fitting_last_layer_runtime_error_clears_state(self) -> None:
        self._assert_runtime_error_clears_state(
            "pt",
            "eval_fitting_last_layer",
            "set_eval_fitting_last_layer_hook",
        )

    def test_pd_eval_descriptor_runtime_error_clears_state(self) -> None:
        self._assert_runtime_error_clears_state(
            "pd",
            "eval_descriptor",
            "set_eval_descriptor_hook",
        )

    def test_pd_eval_fitting_last_layer_runtime_error_clears_state(self) -> None:
        self._assert_runtime_error_clears_state(
            "pd",
            "eval_fitting_last_layer",
            "set_eval_fitting_last_layer_hook",
        )


if __name__ == "__main__":
    unittest.main()
