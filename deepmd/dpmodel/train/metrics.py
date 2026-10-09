# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent accumulation of training display metrics.

Two weighting policies share one accumulator:

* Training steps use unit weight, so the reported value is the arithmetic
  mean of the detached per-step metrics.
* Validation batches use an atom weight (real atoms in the batch), so the
  reported value is the atom-weighted mean of per-atom metrics.

Optional labels emit NaN through ``display_if_exist``. Those observations are
excluded from that metric's sum and weight so a single unlabeled batch cannot
poison an otherwise finite interval. Distributed runs keep local sums and
weights on device and reduce them only at a display boundary.
"""

from __future__ import (
    annotations,
)

from collections.abc import (
    Mapping,
    Sequence,
)
from math import (
    isnan,
)
from typing import (
    Any,
)


def _is_nan(value: Any) -> bool:
    """Return whether a scalar host or array value is NaN."""
    if value is None:
        return True
    try:
        return bool(isnan(float(value)))
    except (TypeError, ValueError):
        return False


class MetricAccumulator:
    """Accumulate weighted sums and counts of detached display metrics.

    Metric names are declared before training so an unsampled task keeps the
    same columns as a sampled task. Each call to :meth:`add` is one
    observation: training uses ``weight=1``, validation passes the number of
    real atoms. A NaN metric is skipped for that observation only.

    Scalar arrays stay on their original device until :meth:`average`
    converts them to Python floats for display. Out-of-place addition
    preserves recorded values when a backend reuses an output buffer.

    Parameters
    ----------
    metric_names : Mapping[str, Sequence[str]]
        Display metric names for each task, excluding internal loss terms.
    """

    def __init__(self, metric_names: Mapping[str, Sequence[str]]) -> None:
        self._names: dict[str, tuple[str, ...]] = {
            task: tuple(sorted(names)) for task, names in metric_names.items()
        }
        self._sums: dict[str, dict[str, Any]] = {
            task: dict.fromkeys(names, 0.0) for task, names in self._names.items()
        }
        self._weights: dict[str, dict[str, Any]] = {
            task: dict.fromkeys(names, 0.0) for task, names in self._names.items()
        }
        self._observations = dict.fromkeys(self._names, 0)

    @property
    def task_keys(self) -> tuple[str, ...]:
        """Return the declared task keys in insertion order."""
        return tuple(self._names)

    def metric_names(self, task_key: str) -> tuple[str, ...]:
        """Return the declared metric names for one task."""
        return self._names[task_key]

    def add(
        self,
        task_key: str,
        metrics: Mapping[str, Any],
        *,
        weight: float = 1.0,
    ) -> None:
        """Record one observation of detached scalar metrics.

        Backends detach metrics from their differentiation graph before
        calling this method. Missing or NaN metrics are omitted from that
        metric's weighted average rather than treated as zero. A non-positive
        weight records the observation for task sampling but contributes to
        no metric totals.
        """
        names = self._names[task_key]
        unknown = metrics.keys() - set(names)
        if unknown:
            raise ValueError(
                f"Undeclared display metrics for task {task_key!r}: {sorted(unknown)}"
            )
        sums = self._sums[task_key]
        weights = self._weights[task_key]
        for name in names:
            value = metrics.get(name, float("nan"))
            if weight <= 0.0 or _is_nan(value):
                continue
            # Out-of-place update: backends may overwrite the tensor buffer
            # that produced ``value`` on the next step.
            sums[name] = sums[name] + value * weight
            weights[name] = weights[name] + weight
        self._observations[task_key] += 1

    def count(self, task_key: str) -> int:
        """Return the number of recorded observations for one task.

        This is the sampling count used to detect tasks that never ran in a
        display interval. It is independent of per-metric weights.
        """
        return self._observations[task_key]

    def average(self, task_key: str) -> dict[str, float]:
        """Return weighted averages, or NaN for every metric with zero weight."""
        sums = self._sums[task_key]
        weights = self._weights[task_key]
        result: dict[str, float] = {}
        for name in self._names[task_key]:
            weight = float(weights[name])
            if weight == 0.0 or _is_nan(weight):
                result[name] = float("nan")
            else:
                result[name] = float(sums[name] / weights[name])
        return result

    def reset(self) -> None:
        """Clear interval values and counts while preserving metric names."""
        for task, names in self._names.items():
            self._sums[task] = dict.fromkeys(names, 0.0)
            self._weights[task] = dict.fromkeys(names, 0.0)
            self._observations[task] = 0

    def flat_state(self) -> list[Any]:
        """Pack sums, weights, and observation counts for a collective reduce.

        Layout is stable across ranks that share the same constructor
        arguments: for each task, all metric sums, then all metric weights,
        then the observation count.
        """
        values: list[Any] = []
        for task, names in self._names.items():
            sums = self._sums[task]
            weights = self._weights[task]
            values.extend(sums[name] for name in names)
            values.extend(weights[name] for name in names)
            values.append(float(self._observations[task]))
        return values

    def load_flat_state(self, values: Sequence[Any]) -> None:
        """Replace local state with a previously packed, possibly reduced vector."""
        expected = sum(2 * len(names) + 1 for names in self._names.values())
        if len(values) != expected:
            raise ValueError(
                f"MetricAccumulator state length {len(values)} != expected {expected}"
            )
        index = 0
        for task, names in self._names.items():
            sums = self._sums[task]
            weights = self._weights[task]
            for name in names:
                sums[name] = values[index]
                index += 1
            for name in names:
                weights[name] = values[index]
                index += 1
            self._observations[task] = int(float(values[index]))
            index += 1


# Historical name kept for call sites and tests that predate the weighted API.
TrainingMetricAccumulator = MetricAccumulator
