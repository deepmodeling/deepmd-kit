# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the training log messages."""

import datetime

import pytest

from deepmd.loggers.training import (
    format_training_message,
    log_parameter_counts,
)

_LOGGER = "deepmd.loggers.training"


def test_progress_message_reports_wall_time_alone() -> None:
    assert (
        format_training_message(batch=100, wall_time=18.41)
        == "Batch     100: total wall time = 18.41 s"
    )


def test_progress_message_appends_the_estimated_finish() -> None:
    message = format_training_message(
        batch=100,
        wall_time=18.41,
        eta=100,
        current_time=datetime.datetime(
            2026, 6, 7, 5, 21, 29, tzinfo=datetime.timezone.utc
        ),
    )

    assert message.startswith("Batch     100: total wall time = 18.41 s, eta = 0:01:40")


def test_single_task_parameter_count_is_reported_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        log_parameter_counts({"Default": (1_500_000, 2_000_000)}, multi_task=False)

    assert caplog.records[-1].message == "Model Params:  2.000 M   (Trainable: 1.500 M)"


def test_multi_task_parameter_counts_are_flagged_as_approximate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        log_parameter_counts(
            {"a": (1_000_000, 1_000_000), "b": (500_000, 500_000)},
            multi_task=True,
        )

    messages = [record.message for record in caplog.records]
    assert "may include duplicates" in messages[0]
    assert messages[1].startswith("Model Params [a]: 1.000 M")
    assert messages[2].startswith("Model Params [b]: 0.500 M")
