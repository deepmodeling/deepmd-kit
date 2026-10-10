# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch profiler lifecycle for pt_expt training.

Owns create/enter, per-step ``profiler.step()``, Chrome/TensorBoard export, and
release on completion or failure. Control stays outside the compiled model
graph. Disabled configurations install no observer.

When both ``enable_profiler`` and ``profiling`` are set, one session serves both
sinks: a custom ``on_trace_ready`` exports the Kineto trace **once**, then places
the file into the TensorBoard log dir and the rank-resolved ``profiling_file``.
``close()`` only performs a Chrome export if that ready-handler save never ran
(short runs that never reach ``RECORD_AND_SAVE``). Profiling-only keeps
end-of-run Chrome export with ``on_trace_ready=None``; enable-profiler-only uses
the TensorBoard handler and never Chrome-exports at close.
"""

from __future__ import (
    annotations,
)

import logging
import os
import shutil
import socket
import time
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.dpmodel.train.observer import (
    StepObservation,
    TrainingObserver,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )

    from deepmd.dpmodel.train.trainer import (
        RankContext,
        TrainingTaskCollection,
    )

log = logging.getLogger(__name__)

__all__ = [
    "TorchProfilerObserver",
    "create_profiler_observer",
    "resolve_chrome_trace_path",
]

# Match legacy PT so enable_profiler / profiling keep their established meaning.
_DEFAULT_SCHEDULE = {"wait": 1, "warmup": 15, "active": 3, "repeat": 1}


def resolve_chrome_trace_path(
    path: str,
    *,
    rank: int,
    world_size: int,
) -> str:
    """Return a collision-free Chrome trace path for the calling rank.

    Single-process training keeps ``path`` unchanged so ``profiling_file`` is
    honored literally. Distributed runs insert a deterministic ``.rankN``
    suffix before the extension.
    """
    if world_size <= 1:
        return path
    target = Path(path)
    return str(target.with_name(f"{target.stem}.rank{rank}{target.suffix}"))


def create_profiler_observer(
    training_params: Mapping[str, Any],
    *,
    rank_context: RankContext,
) -> TorchProfilerObserver | None:
    """Return a profiler observer when enabled; otherwise ``None``.

    A disabled configuration returns ``None`` so the trainer installs no
    observer and never constructs a profiler session.
    """
    enable_profiler = bool(training_params.get("enable_profiler", False))
    profiling = bool(training_params.get("profiling", False))
    if not enable_profiler and not profiling:
        return None
    return TorchProfilerObserver(
        enable_profiler=enable_profiler,
        profiling=profiling,
        profiling_file=str(training_params.get("profiling_file", "timeline.json")),
        tensorboard_log_dir=str(training_params.get("tensorboard_log_dir", "log")),
        rank_context=rank_context,
    )


class TorchProfilerObserver(TrainingObserver):
    """Drive one Torch profiler session across the training observer hooks."""

    def __init__(
        self,
        *,
        enable_profiler: bool,
        profiling: bool,
        profiling_file: str,
        tensorboard_log_dir: str,
        rank_context: RankContext,
        profiler_factory: Callable[..., Any] | None = None,
        schedule_factory: Callable[..., Any] | None = None,
        tensorboard_handler_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._enable_profiler = bool(enable_profiler)
        self._profiling = bool(profiling)
        self._profiling_file = str(profiling_file)
        self._tensorboard_log_dir = str(tensorboard_log_dir)
        self._rank_context = rank_context
        self._profiler_factory = profiler_factory
        self._schedule_factory = schedule_factory
        self._tensorboard_handler_factory = tensorboard_handler_factory
        self._profiler: Any | None = None
        self._closed = False
        # True after a ready-handler (or short-run close fallback) has already
        # called export_chrome_trace once for this session.
        self._trace_saved = False
        self._chrome_trace_path = resolve_chrome_trace_path(
            self._profiling_file,
            rank=rank_context.rank,
            world_size=rank_context.world_size,
        )

    @property
    def enabled(self) -> bool:
        """Whether this observer owns an active profiling session."""
        return self._enable_profiler or self._profiling

    @property
    def chrome_trace_path(self) -> str:
        """Resolved Chrome JSON path for this rank."""
        return self._chrome_trace_path

    @property
    def profiler(self) -> Any | None:
        return self._profiler

    @property
    def trace_saved(self) -> bool:
        """Whether a Chrome/Kineto export has already completed this session."""
        return self._trace_saved

    def wants_step(self, display_step: int) -> bool:
        # Every completed optimizer step must call profiler.step() once.
        return self.enabled

    def on_train_begin(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        if not self.enabled:
            return
        profiler_factory = self._profiler_factory
        schedule_factory = self._schedule_factory
        handler_factory = self._tensorboard_handler_factory
        if (
            profiler_factory is None
            or schedule_factory is None
            or handler_factory is None
        ):
            import torch.profiler as torch_profiler

            if profiler_factory is None:
                profiler_factory = torch_profiler.profile
            if schedule_factory is None:
                schedule_factory = torch_profiler.schedule
            if handler_factory is None:
                handler_factory = torch_profiler.tensorboard_trace_handler

        on_trace_ready = self._build_on_trace_ready(handler_factory)

        self._profiler = profiler_factory(
            schedule=schedule_factory(**_DEFAULT_SCHEDULE),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            with_stack=True,
        )
        self._profiler.start()
        self._closed = False
        self._trace_saved = False
        log.info(
            "Torch profiler started (enable_profiler=%s, profiling=%s)",
            self._enable_profiler,
            self._profiling,
        )

    def on_step_end(self, observation: StepObservation) -> None:
        profiler = self._profiler
        if profiler is None:
            return
        profiler.step()

    def on_train_end(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop the profiler, export Chrome traces when still needed, and release."""
        profiler = self._profiler
        if profiler is None or self._closed:
            self._profiler = None
            self._closed = True
            return
        try:
            # stop() may invoke on_trace_ready for a pending RECORD_AND_SAVE cycle.
            profiler.stop()
            if self._enable_profiler and not self._profiling:
                log.info(
                    "Profiler TensorBoard traces saved under %s",
                    self._tensorboard_log_dir,
                )
            elif self._profiling and not self._enable_profiler:
                # Profiling-only: always end-of-run Chrome export.
                self._ensure_parent_dir(self._chrome_trace_path)
                profiler.export_chrome_trace(self._chrome_trace_path)
                self._trace_saved = True
                log.info(
                    "Profiler Chrome trace saved to: %s",
                    self._chrome_trace_path,
                )
            elif self._enable_profiler and self._profiling:
                if not self._trace_saved:
                    # Short run: schedule never fired ready-handler; one save, both sinks.
                    self._export_combined_trace(profiler)
                else:
                    log.info(
                        "Profiler traces placed under %s and at %s",
                        self._tensorboard_log_dir,
                        self._chrome_trace_path,
                    )
        finally:
            self._profiler = None
            self._closed = True

    def _build_on_trace_ready(
        self,
        handler_factory: Callable[..., Any] | None,
    ) -> Callable[[Any], None] | None:
        """Choose the ready callback for the configured sinks.

        - enable_profiler only → stock TensorBoard handler
        - profiling only → ``None`` (export at close)
        - combined → export once, then place into TB dir and profiling_file
        """
        if self._enable_profiler and self._profiling:
            Path(self._tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
            return self._combined_on_trace_ready
        if self._enable_profiler:
            assert handler_factory is not None
            Path(self._tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
            return handler_factory(
                self._tensorboard_log_dir,
                worker_name=self._tensorboard_worker_name(),
            )
        return None

    def _combined_on_trace_ready(self, prof: Any) -> None:
        """Save the Kineto trace once and fan out to both configured sinks."""
        self._export_combined_trace(prof)

    def _export_combined_trace(self, prof: Any) -> None:
        """Call ``export_chrome_trace`` once; copy into TB log dir and profiling_file."""
        chrome_path = self._chrome_trace_path
        self._ensure_parent_dir(chrome_path)
        Path(self._tensorboard_log_dir).mkdir(parents=True, exist_ok=True)

        # Kineto allows only one export per saved cycle. Write the user-facing
        # profiling_file first, then copy into the TensorBoard log directory
        # with the same naming convention as tensorboard_trace_handler.
        prof.export_chrome_trace(chrome_path)
        tb_path = self._tensorboard_trace_path()
        shutil.copy2(chrome_path, tb_path)
        self._trace_saved = True
        log.info(
            "Profiler Chrome trace saved to: %s (TensorBoard copy: %s)",
            chrome_path,
            tb_path,
        )

    def _tensorboard_worker_name(self) -> str | None:
        if self._rank_context.world_size > 1:
            return f"rank{self._rank_context.rank}"
        return None

    def _tensorboard_trace_path(self) -> str:
        worker = self._tensorboard_worker_name()
        if not worker:
            worker = f"{socket.gethostname()}_{os.getpid()}"
        file_name = f"{worker}.{time.time_ns()}.pt.trace.json"
        return str(Path(self._tensorboard_log_dir) / file_name)

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = Path(path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
