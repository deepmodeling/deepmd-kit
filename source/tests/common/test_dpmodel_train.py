# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.train import (
    change_model_out_bias,
    change_model_out_bias_by_task,
)


class FakeFittingNet:
    def __init__(self) -> None:
        self.input_stats_sample_func = None

    def compute_input_stats(self, sample_func):
        self.input_stats_sample_func = sample_func


class FakeModel:
    def __init__(self, bias: float = 0.0) -> None:
        self.out_bias = np.full((1, 2, 1), bias)
        self.change_calls = []
        self.fitting_net = FakeFittingNet()

    def get_out_bias(self):
        return self.out_bias

    def change_out_bias(self, sample_func, bias_adjust_mode):
        self.change_calls.append((sample_func, bias_adjust_mode))
        self.out_bias = self.out_bias + 1.0

    def get_type_map(self):
        return ["O", "H"]

    def get_fitting_net(self):
        return self.fitting_net


class TestChangeModelOutBias(unittest.TestCase):
    def test_change_model_out_bias_recomputes_input_stats_when_requested(self):
        model = FakeModel()

        def sample_func():
            return [{"atype": np.array([[0, 1]])}]

        returned = change_model_out_bias(
            model,
            sample_func,
            bias_adjust_mode="set-by-statistic",
            recompute_input_stats=True,
        )

        self.assertIs(returned, model)
        self.assertEqual(model.change_calls, [(sample_func, "set-by-statistic")])
        self.assertIs(model.fitting_net.input_stats_sample_func, sample_func)
        np.testing.assert_allclose(model.get_out_bias(), np.ones((1, 2, 1)))

    def test_change_model_out_bias_by_task_updates_all_models(self):
        models = {
            "task_a": FakeModel(0.0),
            "task_b": FakeModel(2.0),
        }
        sample_funcs = {
            "task_a": lambda: [{"atype": np.array([[0]])}],
            "task_b": lambda: [{"atype": np.array([[1]])}],
        }

        returned = change_model_out_bias_by_task(
            models,
            sample_funcs,
            ["task_a", "task_b"],
        )

        self.assertIs(returned, models)
        np.testing.assert_allclose(models["task_a"].get_out_bias(), np.ones((1, 2, 1)))
        np.testing.assert_allclose(
            models["task_b"].get_out_bias(),
            np.full((1, 2, 1), 3.0),
        )
        self.assertEqual(
            models["task_a"].change_calls,
            [(sample_funcs["task_a"], "change-by-statistic")],
        )
        self.assertEqual(
            models["task_b"].change_calls,
            [(sample_funcs["task_b"], "change-by-statistic")],
        )


if __name__ == "__main__":
    unittest.main()
