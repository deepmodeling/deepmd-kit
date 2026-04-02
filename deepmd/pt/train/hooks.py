# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hook system for extensible training callbacks.

This module provides a hook system that allows users to inject custom
logic at various points during the training process without modifying
the core training code.

Example usage:
    >>> class MyHook(TrainingHook):
    ...     def on_step_end(self, step, logs):
    ...         if step % 100 == 0:
    ...             print(f"Step {step}: loss = {logs.get('loss', 'N/A')}")
    >>> trainer.register_hook(MyHook())
"""

from __future__ import (
    annotations,
)

import logging
from abc import (
    ABC,
)
from enum import (
    IntEnum,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
    )

log = logging.getLogger(__name__)


class HookPriority(IntEnum):
    """Priority levels for hook execution order.

    Lower values execute first. Use these to control the order
    in which hooks are called when multiple hooks are registered.
    """

    HIGHEST = 0  # System-critical hooks (e.g., checkpointing)
    HIGH = 10  # Important monitoring hooks
    NORMAL = 20  # Default priority for user hooks
    LOW = 30  # Logging and non-critical hooks
    LOWEST = 40  # Debug and development hooks


class TrainingHook(ABC):
    """Base class for training hooks.

    Subclass this to implement custom callbacks at various points
    during training. All methods are optional - only override the
    ones you need.

    Attributes
    ----------
    priority : HookPriority
        Execution priority of this hook. Lower values execute first.
    """

    priority: HookPriority = HookPriority.NORMAL

    def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
        """Called at the beginning of training.

        Parameters
        ----------
        logs : Mapping[str, Any] | None
            Dictionary of initial values (e.g., start_step).
        """
        pass

    def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
        """Called at the end of training.

        Parameters
        ----------
        logs : Mapping[str, Any] | None
            Dictionary of final training metrics.
        """
        pass

    def on_epoch_begin(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        """Called at the beginning of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        logs : Mapping[str, Any] | None
            Dictionary of metrics from previous epoch.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        """Called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        logs : Mapping[str, Any] | None
            Dictionary of metrics for this epoch.
        """
        pass

    def on_step_begin(self, step: int, logs: Mapping[str, Any] | None = None) -> None:
        """Called at the beginning of each training step.

        Parameters
        ----------
        step : int
            Current step number.
        logs : Mapping[str, Any] | None
            Dictionary of current training state.
        """
        pass

    def on_step_end(self, step: int, logs: Mapping[str, Any] | None = None) -> None:
        """Called at the end of each training step.

        Parameters
        ----------
        step : int
            Current step number.
        logs : Mapping[str, Any] | None
            Dictionary of metrics for this step (loss, lr, etc.).
        """
        pass

    def on_validation_begin(
        self, step: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Called at the beginning of validation.

        Parameters
        ----------
        step : int
            Current step number.
        logs : Mapping[str, Any] | None
            Dictionary of current training state.
        """
        pass

    def on_validation_end(
        self, step: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Called at the end of validation.

        Parameters
        ----------
        step : int
            Current step number.
        logs : Mapping[str, Any] | None
            Dictionary of validation metrics.
        """
        pass

    def on_save_checkpoint(
        self, step: int, checkpoint_path: str, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Called when a checkpoint is saved.

        Parameters
        ----------
        step : int
            Current step number.
        checkpoint_path : str
            Path where checkpoint was saved.
        logs : Mapping[str, Any] | None
            Dictionary of current training state.
        """
        pass


class HookManager:
    """Manages a collection of training hooks.

    This class handles registration and execution of hooks, ensuring
    they are called in priority order and handling any errors gracefully.

    Attributes
    ----------
    hooks : list[TrainingHook]
        List of registered hooks sorted by priority.
    """

    def __init__(self) -> None:
        """Initialize an empty hook manager."""
        self.hooks: list[TrainingHook] = []

    def register(self, hook: TrainingHook) -> None:
        """Register a new hook.

        The hook is inserted in priority order (lower priority values first).

        Parameters
        ----------
        hook : TrainingHook
            The hook instance to register.
        """
        # Insert in priority order
        idx = len(self.hooks)
        for i, existing_hook in enumerate(self.hooks):
            if hook.priority < existing_hook.priority:
                idx = i
                break
        self.hooks.insert(idx, hook)
        log.debug(
            f"Registered hook {hook.__class__.__name__} at priority {hook.priority}"
        )

    def unregister(self, hook: TrainingHook) -> None:
        """Unregister a previously registered hook.

        Parameters
        ----------
        hook : TrainingHook
            The hook instance to unregister.

        Raises
        ------
        ValueError
            If the hook is not found in the registered hooks.
        """
        if hook in self.hooks:
            self.hooks.remove(hook)
            log.debug(f"Unregistered hook {hook.__class__.__name__}")
        else:
            raise ValueError(f"Hook {hook.__class__.__name__} not found")

    def _call_hooks(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Internal method to call a hook method on all registered hooks.

        Parameters
        ----------
        method_name : str
            Name of the hook method to call.
        *args : Any
            Positional arguments to pass to the hook method.
        **kwargs : Any
            Keyword arguments to pass to the hook method.
        """
        for hook in self.hooks:
            try:
                method = getattr(hook, method_name)
                method(*args, **kwargs)
            except Exception as e:
                log.warning(
                    f"Hook {hook.__class__.__name__}.{method_name} failed: {e}",
                    exc_info=True,
                )

    def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
        """Trigger on_train_begin on all hooks."""
        self._call_hooks("on_train_begin", logs)

    def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
        """Trigger on_train_end on all hooks."""
        self._call_hooks("on_train_end", logs)

    def on_epoch_begin(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        """Trigger on_epoch_begin on all hooks."""
        self._call_hooks("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        """Trigger on_epoch_end on all hooks."""
        self._call_hooks("on_epoch_end", epoch, logs)

    def on_step_begin(self, step: int, logs: Mapping[str, Any] | None = None) -> None:
        """Trigger on_step_begin on all hooks."""
        self._call_hooks("on_step_begin", step, logs)

    def on_step_end(self, step: int, logs: Mapping[str, Any] | None = None) -> None:
        """Trigger on_step_end on all hooks."""
        self._call_hooks("on_step_end", step, logs)

    def on_validation_begin(
        self, step: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Trigger on_validation_begin on all hooks."""
        self._call_hooks("on_validation_begin", step, logs)

    def on_validation_end(
        self, step: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Trigger on_validation_end on all hooks."""
        self._call_hooks("on_validation_end", step, logs)

    def on_save_checkpoint(
        self, step: int, checkpoint_path: str, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Trigger on_save_checkpoint on all hooks."""
        self._call_hooks("on_save_checkpoint", step, checkpoint_path, logs)


class TensorBoardHook(TrainingHook):
    """Hook for logging metrics to TensorBoard.

    This hook automatically logs training metrics to TensorBoard
    at specified intervals.

    Attributes
    ----------
    log_dir : str
        Directory for TensorBoard logs.
    log_freq : int
        Frequency of logging (every N steps).
    """

    def __init__(self, log_dir: str = "logs", log_freq: int = 1) -> None:
        """Initialize TensorBoard hook.

        Parameters
        ----------
        log_dir : str
            Directory for TensorBoard logs.
        log_freq : int
            Frequency of logging (every N steps).
        """
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.writer = None
        self._initialized = False

    def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import (
                SummaryWriter,
            )

            self.writer = SummaryWriter(log_dir=self.log_dir)
            self._initialized = True
            log.info(f"TensorBoard logging enabled at {self.log_dir}")
        except ImportError:
            log.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )

    def on_step_end(self, step: int, logs: Mapping[str, Any] | None = None) -> None:
        """Log metrics to TensorBoard."""
        if not self._initialized or self.writer is None:
            return

        if logs is None:
            return

        display_step = step + 1
        if display_step % self.log_freq != 0 and display_step != 1:
            return

        # Log common metrics
        if "loss" in logs:
            self.writer.add_scalar("train/loss", logs["loss"], display_step)
        if "lr" in logs:
            self.writer.add_scalar("train/learning_rate", logs["lr"], display_step)

        # Log task-specific metrics
        for key, value in logs.items():
            if key not in ["loss", "lr"] and isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, display_step)

    def on_validation_end(
        self, step: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Log validation metrics to TensorBoard."""
        if not self._initialized or self.writer is None or logs is None:
            return

        display_step = step + 1
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{key}", value, display_step)

    def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self._initialized = False


class TimingHook(TrainingHook):
    """Hook for tracking and logging training timing statistics.

    Tracks time per step, ETA, and average training speed.
    """

    priority = HookPriority.LOW  # Run after other hooks

    def __init__(self) -> None:
        """Initialize timing hook."""
        self.step_times: list[float] = []
        self.start_time: float | None = None
        self.last_step_time: float | None = None

    def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
        """Reset timing statistics."""
        import time

        self.step_times = []
        self.start_time = time.time()
        self.last_step_time = self.start_time

    def on_step_end(self, step: int, logs: Mapping[str, Any] | None = None) -> None:
        """Record step timing."""
        import time

        if self.last_step_time is not None:
            step_time = time.time() - self.last_step_time
            self.step_times.append(step_time)
            # Keep only last 100 measurements
            if len(self.step_times) > 100:
                self.step_times.pop(0)
        self.last_step_time = time.time()

    def get_average_step_time(self) -> float:
        """Get average step time over last measurements.

        Returns
        -------
        float
            Average step time in seconds, or 0.0 if no measurements.
        """
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times)

    def get_eta_seconds(self, current_step: int, total_steps: int) -> int:
        """Estimate time to completion.

        Parameters
        ----------
        current_step : int
            Current training step.
        total_steps : int
            Total number of training steps.

        Returns
        -------
        int
            Estimated seconds remaining.
        """
        avg_time = self.get_average_step_time()
        remaining_steps = total_steps - current_step
        return int(avg_time * remaining_steps)


class EarlyStoppingHook(TrainingHook):
    """Hook for early stopping based on validation metrics.

    Stops training when a monitored metric has stopped improving.

    Attributes
    ----------
    monitor : str
        Metric name to monitor (e.g., "val_loss").
    patience : int
        Number of steps with no improvement after which training stops.
    mode : str
        One of {"min", "max"}. In "min" mode, training stops when metric
        stops decreasing; in "max" mode, stops when stops increasing.
    """

    priority = HookPriority.HIGH

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
    ) -> None:
        """Initialize early stopping hook.

        Parameters
        ----------
        monitor : str
            Metric name to monitor.
        patience : int
            Number of steps with no improvement before stopping.
        mode : str
            "min" or "max" - whether to minimize or maximize the metric.
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_value: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

        if mode == "min":
            self.is_better = lambda current, best: current < best
            self.best_value = float("inf")
        elif mode == "max":
            self.is_better = lambda current, best: current > best
            self.best_value = float("-inf")
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_validation_end(
        self, step: int, logs: Mapping[str, Any] | None = None
    ) -> None:
        """Check if metric has improved."""
        if logs is None or self.monitor not in logs:
            return

        current_value = logs[self.monitor]
        if not isinstance(current_value, (int, float)):
            return

        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                log.info(
                    f"Early stopping triggered at step {step}. "
                    f"{self.monitor} didn't improve for {self.patience} evaluations."
                )
