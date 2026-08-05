# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the backend-independent training timer."""

import types

import pytest

from deepmd.dpmodel.train import (
    TrainingTimer,
)
from deepmd.dpmodel.train import timing as timing_module


class FakeClock:
    """Monotonic clock advanced explicitly by the test."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Replace the clock the timer reads, leaving the stdlib one untouched."""
    fake = FakeClock()
    monkeypatch.setattr(timing_module, "time", types.SimpleNamespace(monotonic=fake))
    return fake


def test_eta_follows_the_latest_interval(clock: FakeClock) -> None:
    """A run that speeds up is not held back by its slow start.

    A start-up cost paid once -- data preparation, compilation, autotuning --
    would dominate an estimate based on the average since the run began, and
    forecasts nothing at all while it is still being paid.
    """
    timer = TrainingTimer(start_step=0, num_steps=1000, disp_freq=100)

    clock.advance(100.0)
    slow = timer.record(100)
    clock.advance(10.0)
    fast = timer.record(200)

    assert slow.wall_time == pytest.approx(100.0)
    assert slow.eta is None
    assert fast.wall_time == pytest.approx(10.0)
    assert fast.eta == 80


def test_interval_steps_span_the_display_gap(clock: FakeClock) -> None:
    """The first display covers one step, so the next one covers the rest."""
    timer = TrainingTimer(start_step=0, num_steps=1000, disp_freq=100)

    clock.advance(1.0)
    first = timer.record(1)
    clock.advance(99.0)
    second = timer.record(100)

    assert (first.display_step, first.steps) == (1, 1)
    assert (second.display_step, second.steps) == (100, 99)
    # The rate of the second interval, not of the run so far, drives the eta.
    assert second.eta == 900


def test_restart_measures_from_the_resumed_step(clock: FakeClock) -> None:
    timer = TrainingTimer(start_step=500, num_steps=1000, disp_freq=100)

    clock.advance(50.0)
    opening = timer.record(600)
    clock.advance(50.0)
    following = timer.record(700)

    assert (opening.steps, following.steps) == (100, 100)
    # A restart pays the start-up costs again, so only the interval after the
    # opening one forecasts: 0.5 s per step over the 300 steps left.
    assert opening.eta is None
    assert following.eta == 150


def test_average_excludes_the_start_of_the_run(clock: FakeClock) -> None:
    timer = TrainingTimer(start_step=0, num_steps=1000, disp_freq=100)

    clock.advance(100.0)
    timer.record(100)
    clock.advance(10.0)
    timer.record(200)

    # Only the second interval is representative: 10 s over 100 steps.
    assert timer.format_average() == (
        "average training time: 0.1000 s/batch (900 batches excluded)"
    )


def test_short_runs_keep_every_interval(clock: FakeClock) -> None:
    timer = TrainingTimer(start_step=0, num_steps=150, disp_freq=100)

    clock.advance(50.0)
    timer.record(100)

    assert timer.format_average() == (
        "average training time: 0.5000 s/batch (50 batches excluded)"
    )


def test_average_is_absent_without_a_timed_interval(clock: FakeClock) -> None:
    timer = TrainingTimer(start_step=0, num_steps=1000, disp_freq=100)

    assert timer.format_average() is None
