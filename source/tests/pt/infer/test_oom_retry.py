# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from types import (
    SimpleNamespace,
)
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
    RetrySignal,
)


class DummyAutoBatchSize:
    def __init__(self) -> None:
        self.oom_retry_mode = False
        self.modes: list[bool] = []

    def set_oom_retry_mode(self, enable: bool) -> None:
        self.oom_retry_mode = enable
        self.modes.append(enable)


class TestPytorchOOMRetry(unittest.TestCase):
    def _make_backend(self, method_name: str) -> tuple[Any, MagicMock]:
        try:
            from deepmd.pt.infer.deep_eval import (
                DeepEval,
            )
        except ModuleNotFoundError as exc:
            self.skipTest("pt backend dependencies are unavailable: " + str(exc))

        abstract_methods = getattr(DeepEval, "__abstractmethods__", frozenset())
        try:
            DeepEval.__abstractmethods__ = frozenset()
            deep_eval = object.__new__(DeepEval)
        finally:
            DeepEval.__abstractmethods__ = abstract_methods

        model = MagicMock()
        model.eval_descriptor.return_value = np.array([1.0, 2.0, 3.0])
        model.eval_fitting_last_layer.return_value = np.array([4.0, 5.0, 6.0])

        deep_eval.dp = SimpleNamespace(model={"Default": model})
        deep_eval.auto_batch_size = DummyAutoBatchSize()
        return deep_eval, model

    def _assert_retry_clears_hook_between_attempts(
        self,
        method_name: str,
        hook_name: str,
        expected: np.ndarray,
    ) -> None:
        deep_eval, model = self._make_backend(method_name)
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
        method_name: str,
        hook_name: str,
    ) -> None:
        deep_eval, model = self._make_backend(method_name)
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

    def test_eval_descriptor_retry_clears_hook_between_attempts(self) -> None:
        self._assert_retry_clears_hook_between_attempts(
            "eval_descriptor",
            "set_eval_descriptor_hook",
            np.array([1.0, 2.0, 3.0]),
        )

    def test_eval_fitting_last_layer_retry_clears_hook_between_attempts(
        self,
    ) -> None:
        self._assert_retry_clears_hook_between_attempts(
            "eval_fitting_last_layer",
            "set_eval_fitting_last_layer_hook",
            np.array([4.0, 5.0, 6.0]),
        )

    def test_eval_descriptor_runtime_error_clears_state(self) -> None:
        self._assert_runtime_error_clears_state(
            "eval_descriptor",
            "set_eval_descriptor_hook",
        )

    def test_eval_fitting_last_layer_runtime_error_clears_state(self) -> None:
        self._assert_runtime_error_clears_state(
            "eval_fitting_last_layer",
            "set_eval_fitting_last_layer_hook",
        )


if __name__ == "__main__":
    unittest.main()
