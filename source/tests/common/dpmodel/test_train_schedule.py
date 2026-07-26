# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the backend-independent training-step schedule."""

import numpy as np
import pytest

from deepmd.dpmodel.train import (
    resolve_step_schedule,
)


class FakeData:
    """Training-data stand-in reporting a system count."""

    def __init__(self, nsystems: int) -> None:
        self._nsystems = nsystems

    def get_nsystems(self) -> int:
        return self._nsystems


def _unreachable_epoch_length(model_key: str) -> int:
    raise AssertionError(f"epoch length of '{model_key}' must not be needed")


def test_explicit_steps_bypass_the_data() -> None:
    schedule = resolve_step_schedule(
        {"numb_steps": 7},
        multi_task=False,
        model_keys=["Default"],
        training_data={"Default": FakeData(1)},
        epoch_length=_unreachable_epoch_length,
    )

    assert schedule.num_steps == 7
    assert schedule.model_prob is None


@pytest.mark.parametrize(
    ("num_epoch", "expected"),
    [(1.0, 40), (2.5, 100), (0.25, 10), (1.0 / 3.0, 14)],
)
def test_single_task_epochs_round_up(num_epoch: float, expected: int) -> None:
    schedule = resolve_step_schedule(
        {"numb_epoch": num_epoch},
        multi_task=False,
        model_keys=["Default"],
        training_data={"Default": FakeData(1)},
        epoch_length=lambda _: 40,
    )

    assert schedule.num_steps == expected
    assert schedule.model_prob is None


def test_single_task_requires_a_run_length() -> None:
    with pytest.raises(ValueError, match=r"numb_steps or training\.num_epoch"):
        resolve_step_schedule(
            {},
            multi_task=False,
            model_keys=["Default"],
            training_data={"Default": FakeData(1)},
            epoch_length=_unreachable_epoch_length,
        )


def test_single_task_rejects_non_positive_epochs() -> None:
    with pytest.raises(ValueError, match="num_epoch must be positive"):
        resolve_step_schedule(
            {"numb_epoch": 0.0},
            multi_task=False,
            model_keys=["Default"],
            training_data={"Default": FakeData(1)},
            epoch_length=_unreachable_epoch_length,
        )


def test_empty_training_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive for task 'Default'"):
        resolve_step_schedule(
            {"numb_epoch": 1.0},
            multi_task=False,
            model_keys=["Default"],
            training_data={"Default": FakeData(1)},
            epoch_length=lambda _: 0,
        )


def test_epoch_lengths_are_pinned_to_rank_zero() -> None:
    """A rank derives its run length from the broadcast value, not its own."""
    schedule = resolve_step_schedule(
        {"numb_epoch": 2.0},
        multi_task=False,
        model_keys=["Default"],
        training_data={"Default": FakeData(1)},
        epoch_length=lambda _: 41,
        broadcast=lambda lengths: [40] * len(lengths),
    )

    assert schedule.num_steps == 80


def test_multi_task_epoch_lengths_are_pinned_to_rank_zero() -> None:
    """Every task's epoch length comes from the broadcast, not from this rank."""
    schedule = resolve_step_schedule(
        {"num_epoch_dict": {"model_1": 1.0, "model_2": 4.0}},
        multi_task=True,
        model_keys=["model_1", "model_2"],
        training_data={"model_1": FakeData(1), "model_2": FakeData(1)},
        epoch_length=lambda _: 11,
        broadcast=lambda lengths: [40, 10][: len(lengths)],
    )

    assert schedule.num_steps == 80
    np.testing.assert_allclose(schedule.model_prob, [0.5, 0.5])


def test_multi_task_epoch_dict_splits_steps_by_epoch_target() -> None:
    """Each task receives the steps its epoch target asks for."""
    epoch_lengths = {"model_1": 40, "model_2": 10}
    num_epoch_dict = {"model_1": 1.0, "model_2": 4.0}

    schedule = resolve_step_schedule(
        {"num_epoch_dict": num_epoch_dict},
        multi_task=True,
        model_keys=["model_1", "model_2"],
        training_data={key: FakeData(1) for key in epoch_lengths},
        epoch_length=epoch_lengths.__getitem__,
    )

    assert schedule.num_steps == 80
    np.testing.assert_allclose(schedule.model_prob, [0.5, 0.5])
    for index, model_key in enumerate(["model_1", "model_2"]):
        expected_epochs = num_epoch_dict[model_key]
        drawn_steps = schedule.num_steps * schedule.model_prob[index]
        assert drawn_steps / epoch_lengths[model_key] == pytest.approx(expected_epochs)


def test_multi_task_epoch_dict_rejects_explicit_steps() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_step_schedule(
            {"numb_steps": 10, "num_epoch_dict": {"model_1": 1.0}},
            multi_task=True,
            model_keys=["model_1"],
            training_data={"model_1": FakeData(1)},
            epoch_length=lambda _: 40,
        )


def test_multi_task_steps_keep_configured_model_prob() -> None:
    schedule = resolve_step_schedule(
        {"numb_steps": 12, "model_prob": {"model_1": 3.0, "model_2": 1.0}},
        multi_task=True,
        model_keys=["model_1", "model_2"],
        training_data={"model_1": FakeData(1), "model_2": FakeData(1)},
        epoch_length=_unreachable_epoch_length,
    )

    assert schedule.num_steps == 12
    np.testing.assert_allclose(schedule.model_prob, [0.75, 0.25])


def test_multi_task_requires_a_run_length() -> None:
    with pytest.raises(ValueError, match="num_epoch_dict must be set"):
        resolve_step_schedule(
            {},
            multi_task=True,
            model_keys=["model_1"],
            training_data={"model_1": FakeData(1)},
            epoch_length=_unreachable_epoch_length,
        )
