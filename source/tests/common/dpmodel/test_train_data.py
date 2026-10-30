# SPDX-License-Identifier: LGPL-3.0-or-later
from unittest.mock import (
    Mock,
)

import pytest

from deepmd.dpmodel.train.data import (
    _print_summary,
    iter_training_task_configs,
    make_task_maps,
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


def test_make_task_maps_closes_completed_tasks_on_factory_failure() -> None:
    """A later task failure releases already-acquired training and validation data."""
    config = {
        "model": {"model_dict": {"first": {}, "second": {}}},
        "training": {
            "data_dict": {
                "first": {"training_data": {}},
                "second": {"training_data": {}},
            }
        },
    }
    training, validation = Mock(), Mock()
    factory = Mock(
        side_effect=[(training, validation, None), ValueError("second task failed")]
    )
    with pytest.raises(ValueError, match="second task failed"):
        make_task_maps(config, factory)
    training.close.assert_called_once_with()
    validation.close.assert_called_once_with()


def test_make_task_maps_transfers_successful_data_to_caller() -> None:
    """Successful construction keeps data open for the caller's training loop."""
    config = {"model": {}, "training": {"training_data": {}}}
    training, validation = Mock(), Mock()
    maps = make_task_maps(config, Mock(return_value=(training, validation, None)))
    assert list(maps[0].values()) == [training]
    assert list(maps[1].values()) == [validation]
    training.close.assert_not_called()
    validation.close.assert_not_called()
