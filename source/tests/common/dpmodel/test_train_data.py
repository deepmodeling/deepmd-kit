# SPDX-License-Identifier: LGPL-3.0-or-later
from unittest.mock import (
    Mock,
)

import numpy as np
import pytest

from deepmd.dpmodel.train.data import (
    _print_summary,
    iter_training_task_configs,
    make_task_maps,
    next_training_batch,
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


@pytest.mark.parametrize("valid", [False, True])
def test_training_filter_retries_empty_batches_and_preserves_ragged_fields(
    valid: bool,
) -> None:
    class Data:
        def __init__(self) -> None:
            self.calls = 0

        def get_batch(self) -> dict[str, np.ndarray]:
            self.calls += 1
            return {
                "atype": np.array([0, 1, 1]),
                "coord": np.arange(9).reshape(3, 3),
                "force": np.arange(9).reshape(3, 3),
                "n_node": np.array([2, 1]),
                "energy": np.array([[7.0], [0.5]]),
                "find_pair_margin": np.asarray(True),
                "pair_margin": np.array(
                    [[0.2], [1.2 if valid and self.calls == 2 else 0.2]]
                ),
            }

        def get_batch_pass_length(self) -> int:
            return 2

    data = Data()
    if not valid:
        with pytest.raises(RuntimeError, match="No training frame of task 'test'"):
            next_training_batch(data, filter_enabled=True, task_key="test")
        assert data.calls == 2
        return
    batch = next_training_batch(data, filter_enabled=True, task_key="test")
    assert data.calls == 2
    np.testing.assert_array_equal(batch["n_node"], [1])
    np.testing.assert_array_equal(batch["atype"], [1])
    np.testing.assert_array_equal(batch["force"], [[6, 7, 8]])
    np.testing.assert_array_equal(batch["energy"], [[0.5]])
