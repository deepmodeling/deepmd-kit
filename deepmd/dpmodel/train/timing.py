# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wall-clock accounting of a training run.

Progress is reported once per display interval and summarized once at the end
of the run. Both numbers come from the same bookkeeping, which is independent
of the backend performing the steps.
"""

from __future__ import (
    annotations,
)

import datetime
import time
from dataclasses import (
    dataclass,
)

__all__ = ["DisplayInterval", "TrainingTimer"]


@dataclass(frozen=True)
class DisplayInterval:
    """Wall-clock summary of the steps since the previous display.

    Attributes
    ----------
    display_step : int
        One-based index of the step the interval ends at.
    wall_time : float
        Time elapsed since the previous display, in s.
    steps : int
        Number of training steps performed during the interval.
    eta : int or None
        Estimated time left in the run, in s. ``None`` for the opening
        interval, whose rate carries the one-off costs of starting the run and
        so forecasts nothing.
    timestamp : datetime.datetime
        Local time at the end of the interval.
    """

    display_step: int
    wall_time: float
    steps: int
    eta: int | None
    timestamp: datetime.datetime


class TrainingTimer:
    """Timer producing per-interval progress and a run-level average.

    The remaining time is extrapolated from the rate of the interval that just
    ended rather than from the average since the run began, because the start
    of a run also carries one-off costs -- data preparation, graph compilation,
    autotuning -- that never recur and would otherwise inflate every estimate.

    The reported average deliberately omits the first ``disp_freq`` steps for
    the same reason, unless the run is too short for the exclusion to leave a
    meaningful sample.

    Parameters
    ----------
    start_step : int
        Step the run starts from, non-zero when restarting.
    num_steps : int
        Step the run ends at.
    disp_freq : int
        Number of steps between two displays.
    """

    def __init__(self, *, start_step: int, num_steps: int, disp_freq: int) -> None:
        self._start_step = int(start_step)
        self._num_steps = int(num_steps)
        self._disp_freq = max(1, int(disp_freq))
        self._interval_start = time.monotonic()
        self._last_display_step = self._start_step
        self._timed_time = 0.0
        self._timed_steps = 0
        self._recorded_any = False

    def record(self, display_step: int) -> DisplayInterval:
        """Close the current interval and open the next one.

        Parameters
        ----------
        display_step : int
            One-based index of the step that just completed.

        Returns
        -------
        DisplayInterval
            Wall-clock summary of the interval that just ended.
        """
        interval_end = time.monotonic()
        wall_time = interval_end - self._interval_start
        steps = max(1, display_step - self._last_display_step)
        self._interval_start = interval_end
        self._last_display_step = display_step
        if self._counts_toward_average(display_step):
            self._timed_time += wall_time
            self._timed_steps += steps
        # The opening interval absorbs the one-off costs of the run -- data
        # preparation, graph compilation, autotuning -- and its rate would
        # forecast a run many times longer than the real one.
        forecasts = self._recorded_any
        self._recorded_any = True
        return DisplayInterval(
            display_step=display_step,
            wall_time=wall_time,
            steps=steps,
            eta=int((self._num_steps - display_step) * wall_time / steps)
            if forecasts
            else None,
            timestamp=datetime.datetime.now(datetime.timezone.utc).astimezone(),
        )

    def format_average(self) -> str | None:
        """Report the average step time of the run.

        Returns
        -------
        str or None
            The average time per step over the representative intervals, or
            ``None`` when no interval was representative.
        """
        if not self._timed_steps:
            return None
        message = (
            f"average training time: {self._timed_time / self._timed_steps:.4f} s/batch"
        )
        excluded = self._num_steps - self._start_step - self._timed_steps
        if excluded > 0:
            message += f" ({excluded} batches excluded)"
        return message

    def _counts_toward_average(self, display_step: int) -> bool:
        """Whether an interval ending at ``display_step`` is representative."""
        if self._num_steps - self._start_step <= 2 * self._disp_freq:
            return True
        return display_step - 1 - self._start_step >= self._disp_freq
