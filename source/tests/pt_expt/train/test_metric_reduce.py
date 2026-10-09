# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Torch metric-window reduction helpers."""

import pytest

torch = pytest.importorskip("torch")

from deepmd.dpmodel.train.metrics import (  # noqa: E402
    MetricAccumulator,
)
from deepmd.pt_expt.train.metrics import (  # noqa: E402
    all_reduce_metric_accumulator,
)


def test_all_reduce_is_noop_without_process_group() -> None:
    accumulator = MetricAccumulator({"task": ("rmse",)})
    accumulator.add("task", {"rmse": 1.5})
    all_reduce_metric_accumulator(accumulator)
    assert accumulator.average("task") == {"rmse": 1.5}
