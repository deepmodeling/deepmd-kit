# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for backend-independent training metric windows."""

import math

import numpy as np
import pytest

from deepmd.dpmodel.train.metrics import (
    TrainingMetricAccumulator,
)


def test_tasks_have_independent_counts_and_stable_columns() -> None:
    accumulator = TrainingMetricAccumulator(
        {"energy": ("rmse_f", "rmse"), "property": ("mae",)}
    )
    assert list(accumulator.average("energy")) == ["rmse", "rmse_f"]
    assert math.isnan(accumulator.average("property")["mae"])
    accumulator.add("energy", {"rmse": 1.0, "rmse_f": 2.0})
    accumulator.add("energy", {"rmse": 3.0, "rmse_f": 8.0})
    accumulator.add("property", {"mae": 7.0})
    assert accumulator.average("energy") == {"rmse": 2.0, "rmse_f": 5.0}
    assert accumulator.average("property") == {"mae": 7.0}
    assert accumulator.count("energy") == 2
    assert accumulator.count("property") == 1

    accumulator.reset()
    assert accumulator.count("energy") == 0
    assert accumulator.count("property") == 0
    assert list(accumulator.average("energy")) == ["rmse", "rmse_f"]
    assert all(math.isnan(value) for value in accumulator.average("energy").values())
    accumulator.add("energy", {"rmse": 9.0, "rmse_f": 4.0})
    assert accumulator.average("energy") == {"rmse": 9.0, "rmse_f": 4.0}


def test_accumulation_does_not_alias_or_modify_backend_buffers() -> None:
    accumulator = TrainingMetricAccumulator({"task": ("rmse",)})
    value = np.array(2.0, dtype=np.float32)
    accumulator.add("task", {"rmse": value})
    value[...] = 6.0
    accumulator.add("task", {"rmse": value})
    assert value == 6.0
    assert accumulator.average("task") == {"rmse": 4.0}


@pytest.mark.parametrize("metrics", [{}, {"rmse": float("nan")}])
def test_missing_and_nan_metrics_are_not_silently_zeroed(metrics: dict) -> None:
    accumulator = TrainingMetricAccumulator({"task": ("rmse",)})
    accumulator.add("task", {"rmse": 1.0})
    accumulator.add("task", metrics)
    assert math.isnan(accumulator.average("task")["rmse"])
    assert accumulator.count("task") == 2


def test_unknown_metric_does_not_partially_update_the_window() -> None:
    accumulator = TrainingMetricAccumulator({"task": ("rmse",)})
    with pytest.raises(ValueError, match=r"Undeclared display metrics.*mae"):
        accumulator.add("task", {"rmse": 1.0, "mae": 2.0})
    assert accumulator.count("task") == 0
