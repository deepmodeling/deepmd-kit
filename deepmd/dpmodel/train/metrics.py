# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent accumulation of training display metrics."""

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import (
    Any,
)


class TrainingMetricAccumulator:
    """Average per-step metrics independently for each training task.

    Metric names are declared before training so an unsampled task has the
    same columns as a sampled task. Each update has equal weight, matching
    the arithmetic mean of the reported per-step metrics rather than a
    pooled error over atoms. Scalar arrays stay on their original device
    until :meth:`average` converts them to Python floats for display.

    Parameters
    ----------
    metric_names : Mapping[str, Sequence[str]]
        Display metric names for each task, excluding internal loss terms.
    """

    def __init__(self, metric_names: Mapping[str, Sequence[str]]) -> None:
        self._totals: dict[str, dict[str, Any]] = {
            task: dict.fromkeys(sorted(names), 0.0)
            for task, names in metric_names.items()
        }
        self._counts = dict.fromkeys(metric_names, 0)

    def add(self, task_key: str, metrics: Mapping[str, Any]) -> None:
        """Record one optimizer step's detached scalar metrics.

        Backends detach metrics from their differentiation graph before
        calling this method. Missing metrics propagate NaN rather than being
        treated as zero. Out-of-place addition preserves the recorded value
        when a backend reuses an output buffer on its next step.
        """
        totals = self._totals[task_key]
        unknown = metrics.keys() - totals.keys()
        if unknown:
            raise ValueError(
                f"Undeclared display metrics for task {task_key!r}: {sorted(unknown)}"
            )
        for name, total in totals.items():
            totals[name] = total + metrics.get(name, float("nan"))
        self._counts[task_key] += 1

    def count(self, task_key: str) -> int:
        """Return the number of recorded optimizer steps for one task."""
        return self._counts[task_key]

    def average(self, task_key: str) -> dict[str, float]:
        """Return interval averages, or NaN for every unsampled metric."""
        totals = self._totals[task_key]
        count = self._counts[task_key]
        if count == 0:
            return dict.fromkeys(totals, float("nan"))
        return {name: float(total / count) for name, total in totals.items()}

    def reset(self) -> None:
        """Clear interval values and counts while preserving metric names."""
        for task, totals in self._totals.items():
            self._totals[task] = dict.fromkeys(totals, 0.0)
            self._counts[task] = 0
