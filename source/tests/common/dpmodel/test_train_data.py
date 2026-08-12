# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from deepmd.dpmodel.train.data import (
    _print_summary,
    iter_training_task_configs,
)
from deepmd.utils.stat_file import (
    StatFileSpec,
)


def test_print_summary_supports_legacy_no_probability_signature() -> None:
    class LegacySummary:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def print_summary(self, name: str) -> None:
            self.calls.append(name)

    data = LegacySummary()

    _print_summary(data, "training", [1.0])

    assert data.calls == ["training"]


def test_print_summary_does_not_swallow_internal_type_error() -> None:
    class BrokenSummary:
        def print_summary(self, name: str, prob: list[float] | None) -> None:
            raise TypeError("internal summary failure")

    with pytest.raises(TypeError, match="internal summary failure"):
        _print_summary(BrokenSummary(), "training", [1.0])


def test_training_task_config_preserves_stat_file_mode() -> None:
    config = {
        "model": {},
        "training": {
            "training_data": {},
            "stat_file": "stat.hdf5",
            "stat_file_mode": "read",
        },
    }

    task = next(iter_training_task_configs(config))

    assert task.stat_file_spec == StatFileSpec("stat.hdf5", "read")
