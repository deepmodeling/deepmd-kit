# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training logging utilities for output formatting and file I/O.

This module provides clean interfaces for logging training progress,
managing log files, and formatting output messages.
"""

from __future__ import (
    annotations,
)

import logging
from pathlib import (
    Path,
)
from typing import (
    Any,
    Self,
)

from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)

log = logging.getLogger(__name__)


class TrainingLogger:
    """Handles training log output to console and file.

    This class manages the training log file, formats output messages,
    and handles both single-task and multi-task logging scenarios.

    Attributes
    ----------
    log_file : Path | None
        Path to the log file.
    should_print_header : bool
        Whether header needs to be printed.
    is_multitask : bool
        Whether logging for multi-task training.
    model_keys : list[str] | None
        Model keys for multi-task logging.
    """

    def __init__(
        self,
        log_file: str,
        is_multitask: bool = False,
        model_keys: list[str] | None = None,
        rank: int = 0,
        restart: bool = False,
    ) -> None:
        """Initialize training logger.

        Parameters
        ----------
        log_file : str
            Path to log file.
        is_multitask : bool
            Whether this is multi-task training.
        model_keys : list[str] | None
            Model keys for multi-task.
        rank : int
            Process rank (only rank 0 writes to file).
        restart : bool
            Whether this is a restart (append mode).
        """
        self.is_multitask = is_multitask
        self.model_keys = model_keys or []
        self.rank = rank
        self.should_print_header = True

        # Open file only on rank 0
        if rank == 0:
            self.log_path = Path(log_file)
            mode = "a" if restart else "w"
            self._file_handle = open(self.log_path, mode=mode, buffering=1)
        else:
            self.log_path = None
            self._file_handle = None

    def log_step(
        self,
        step: int,
        train_results: dict[str, Any],
        valid_results: dict[str, Any] | dict[str, dict[str, Any]] | None,
        lr: float,
        wall_time: float | None = None,
        eta: int | None = None,
        task_key: str | None = None,
    ) -> None:
        """Log a training step.

        Parameters
        ----------
        step : int
            Current step number.
        train_results : dict[str, Any]
            Training metrics.
        valid_results : dict[str, Any] | dict[str, dict[str, Any]] | None
            Validation metrics.
        lr : float
            Current learning rate.
        wall_time : float | None
            Wall time for step (optional).
        eta : int | None
            Estimated time to completion (optional).
        task_key : str | None
            Current task key for multi-task.
        """
        if self.rank != 0:
            return

        # Print header if needed
        if self.should_print_header and self._file_handle:
            self._print_header(train_results, valid_results)
            self.should_print_header = False

        # Log to console
        self._log_console(
            step, train_results, valid_results, lr, wall_time, eta, task_key
        )

        # Log to file
        if self._file_handle:
            self._print_to_file(step, lr, train_results, valid_results)

    def _log_console(
        self,
        step: int,
        train_results: dict[str, Any],
        valid_results: dict[str, Any] | dict[str, dict[str, Any]] | None,
        lr: float,
        wall_time: float | None,
        eta: int | None,
        task_key: str | None,
    ) -> None:
        """Log to console."""
        if self.is_multitask:
            # Log all tasks
            for key in self.model_keys:
                train_res = (
                    train_results.get(key, {})
                    if isinstance(train_results, dict)
                    else {}
                )
                valid_res = (
                    valid_results.get(key, {})
                    if isinstance(valid_results, dict)
                    else {}
                )

                if train_res:
                    log.info(
                        format_training_message_per_task(
                            batch=step,
                            task_name=f"{key}_trn",
                            rmse=train_res,
                            learning_rate=lr if key == task_key else None,
                        )
                    )

                if valid_res:
                    log.info(
                        format_training_message_per_task(
                            batch=step,
                            task_name=f"{key}_val",
                            rmse=valid_res,
                            learning_rate=None,
                        )
                    )
        else:
            if train_results:
                log.info(
                    format_training_message_per_task(
                        batch=step,
                        task_name="trn",
                        rmse=train_results,
                        learning_rate=lr,
                    )
                )

            if valid_results and isinstance(valid_results, dict):
                log.info(
                    format_training_message_per_task(
                        batch=step,
                        task_name="val",
                        rmse=valid_results,
                        learning_rate=None,
                    )
                )

        # Log timing
        if wall_time is not None and eta is not None:
            log.info(
                format_training_message(
                    batch=step,
                    wall_time=wall_time,
                    eta=eta,
                )
            )

    def _print_header(
        self,
        train_results: dict[str, Any],
        valid_results: dict[str, Any] | dict[str, dict[str, Any]] | None,
    ) -> None:
        """Print header to log file."""
        if not self._file_handle:
            return

        header = "# {:5s}".format("step")

        if self.is_multitask:
            for key in self.model_keys:
                train_keys = (
                    sorted(train_results.get(key, {}).keys())
                    if isinstance(train_results, dict)
                    else []
                )
                if valid_results and key in (valid_results or {}):
                    for k in train_keys:
                        header += f"   {k + f'_val_{key}':11s} {k + f'_trn_{key}':11s}"
                else:
                    for k in train_keys:
                        header += f"   {k + f'_trn_{key}':11s}"
        else:
            train_keys = sorted(train_results.keys())
            if valid_results:
                for k in train_keys:
                    header += f"   {k + '_val':11s} {k + '_trn':11s}"
            else:
                for k in train_keys:
                    header += f"   {k + '_trn':11s}"

        header += "   {:8s}\n".format("lr")
        header += "# If there is no available reference data, rmse_*_{val,trn} will print nan\n"

        self._file_handle.write(header)
        self._file_handle.flush()

    def _print_to_file(
        self,
        step: int,
        lr: float,
        train_results: dict[str, Any],
        valid_results: dict[str, Any] | dict[str, dict[str, Any]] | None,
    ) -> None:
        """Print formatted line to log file."""
        if not self._file_handle:
            return

        line = f"{step:7d}"

        if self.is_multitask:
            for key in self.model_keys:
                train_res = (
                    train_results.get(key, {})
                    if isinstance(train_results, dict)
                    else {}
                )
                valid_res = (
                    valid_results.get(key, {})
                    if isinstance(valid_results, dict)
                    else {}
                )

                if valid_res:
                    for k in sorted(train_res.keys()):
                        line += f"   {valid_res.get(k, 0.0):11.2e} {train_res.get(k, 0.0):11.2e}"
                else:
                    for k in sorted(train_res.keys()):
                        line += f"   {train_res.get(k, 0.0):11.2e}"
        else:
            train_keys = sorted(train_results.keys())
            if valid_results and isinstance(valid_results, dict):
                for k in train_keys:
                    line += f"   {valid_results.get(k, 0.0):11.2e} {train_results.get(k, 0.0):11.2e}"
            else:
                for k in train_keys:
                    line += f"   {train_results.get(k, 0.0):11.2e}"

        line += f"   {lr:8.1e}\n"
        self._file_handle.write(line)
        self._file_handle.flush()

    def log_summary(
        self, total_time: float, timed_steps: int, excluded_steps: int
    ) -> None:
        """Log training summary.

        Parameters
        ----------
        total_time : float
            Total training time.
        timed_steps : int
            Number of timed steps.
        excluded_steps : int
            Number of excluded steps.
        """
        if timed_steps > 0:
            avg_time = total_time / timed_steps
            msg = f"Average training time: {avg_time:.4f} s/batch"
            if excluded_steps > 0:
                msg += f" ({excluded_steps} batches excluded)"
            log.info(msg)

    def close(self) -> None:
        """Close log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()


class LossAccumulator:
    """Accumulates loss values over multiple steps for averaging.

    This class handles loss accumulation for both single-task and
    multi-task training scenarios.
    """

    def __init__(
        self, is_multitask: bool = False, model_keys: list[str] | None = None
    ) -> None:
        """Initialize loss accumulator.

        Parameters
        ----------
        is_multitask : bool
            Whether this is multi-task training.
        model_keys : list[str] | None
            Model keys for multi-task.
        """
        self.is_multitask = is_multitask
        self.model_keys = model_keys or []
        self.reset()

    def reset(self) -> None:
        """Reset accumulated losses."""
        if self.is_multitask:
            self.accumulated: dict[str, dict[str, float]] = {
                key: {} for key in self.model_keys
            }
            self.step_counts: dict[str, int] = dict.fromkeys(self.model_keys, 0)
        else:
            self.accumulated: dict[str, float] = {}
            self.step_count = 0

    def update(
        self,
        more_loss: dict[str, Any],
        task_key: str = "Default",
    ) -> None:
        """Update accumulated losses.

        Parameters
        ----------
        more_loss : dict[str, Any]
            Loss dictionary from step.
        task_key : str
            Task key for multi-task.
        """
        if self.is_multitask:
            self.step_counts[task_key] = self.step_counts.get(task_key, 0) + 1
            if task_key not in self.accumulated:
                self.accumulated[task_key] = {}

            for key, value in more_loss.items():
                if "l2_" in key:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                if key not in self.accumulated[task_key]:
                    self.accumulated[task_key][key] = 0.0
                self.accumulated[task_key][key] += float(value)
        else:
            self.step_count += 1
            for key, value in more_loss.items():
                if "l2_" in key:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                if key not in self.accumulated:
                    self.accumulated[key] = 0.0
                self.accumulated[key] += float(value)

    def get_averaged(self, task_key: str = "Default") -> dict[str, float]:
        """Get averaged losses.

        Parameters
        ----------
        task_key : str
            Task key for multi-task.

        Returns
        -------
        dict[str, float]
            Averaged loss values.
        """
        if self.is_multitask:
            if task_key not in self.accumulated:
                return {}
            count = self.step_counts.get(task_key, 1)
            return {k: v / count for k, v in self.accumulated[task_key].items()}
        else:
            if self.step_count == 0:
                return {}
            return {k: v / self.step_count for k, v in self.accumulated.items()}

    def get_all_averaged(self) -> dict[str, dict[str, float]] | dict[str, float]:
        """Get all averaged losses.

        Returns
        -------
        dict[str, dict[str, float]] | dict[str, float]
            All averaged losses.
        """
        if self.is_multitask:
            return {key: self.get_averaged(key) for key in self.model_keys}
        return self.get_averaged()
