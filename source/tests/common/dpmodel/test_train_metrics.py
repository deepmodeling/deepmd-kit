# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for backend-independent training metric windows."""

import math

import numpy as np
import pytest

from deepmd.dpmodel.train.metrics import (
    MetricAccumulator,
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
def test_missing_and_nan_metrics_are_excluded_from_the_average(metrics: dict) -> None:
    """Optional labels must not poison an otherwise finite interval average."""
    accumulator = MetricAccumulator({"task": ("rmse",)})
    accumulator.add("task", {"rmse": 1.0})
    accumulator.add("task", metrics)
    assert accumulator.average("task") == {"rmse": 1.0}
    assert accumulator.count("task") == 2


def test_metric_with_only_nan_observations_reports_nan() -> None:
    accumulator = MetricAccumulator({"task": ("rmse", "mae")})
    accumulator.add("task", {"rmse": float("nan"), "mae": 2.0})
    accumulator.add("task", {"rmse": float("nan"), "mae": 4.0})
    assert math.isnan(accumulator.average("task")["rmse"])
    assert accumulator.average("task")["mae"] == 3.0


def test_atom_weights_compute_validation_style_averages() -> None:
    accumulator = MetricAccumulator({"task": ("rmse",)})
    accumulator.add("task", {"rmse": 1.0}, weight=2.0)
    accumulator.add("task", {"rmse": 4.0}, weight=1.0)
    assert accumulator.average("task") == {"rmse": 2.0}
    assert accumulator.count("task") == 2


def test_non_positive_weight_skips_metric_totals() -> None:
    accumulator = MetricAccumulator({"task": ("rmse",)})
    accumulator.add("task", {"rmse": 1.0}, weight=0.0)
    assert accumulator.count("task") == 1
    assert math.isnan(accumulator.average("task")["rmse"])


def test_flat_state_round_trip_supports_distributed_reduce() -> None:
    left = MetricAccumulator({"a": ("rmse",), "b": ("mae",)})
    right = MetricAccumulator({"a": ("rmse",), "b": ("mae",)})
    left.add("a", {"rmse": 1.0})
    left.add("b", {"mae": 2.0})
    right.add("a", {"rmse": 3.0})
    # b unsampled on the right rank
    merged = [x + y for x, y in zip(left.flat_state(), right.flat_state())]
    left.load_flat_state(merged)
    assert left.average("a") == {"rmse": 2.0}
    assert left.average("b") == {"mae": 2.0}
    assert left.count("a") == 2
    assert left.count("b") == 1


def test_unknown_metric_does_not_partially_update_the_window() -> None:
    accumulator = TrainingMetricAccumulator({"task": ("rmse",)})
    with pytest.raises(ValueError, match=r"Undeclared display metrics.*mae"):
        accumulator.add("task", {"rmse": 1.0, "mae": 2.0})
    assert accumulator.count("task") == 0
