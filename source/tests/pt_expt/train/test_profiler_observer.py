# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)
from unittest.mock import (
    MagicMock,
)

from deepmd.dpmodel.train.observer import (
    StepObservation,
)
from deepmd.dpmodel.train.trainer import (
    RankContext,
    TrainingTaskCollection,
    TrainStepResult,
)
from deepmd.pt_expt.train.profiler import (
    TorchProfilerObserver,
    create_profiler_observer,
    resolve_chrome_trace_path,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )


def _step(
    display_step: int, *, rank_context: RankContext | None = None
) -> StepObservation:
    ctx = rank_context or RankContext()
    return StepObservation(
        step=display_step - 1,
        display_step=display_step,
        task_key="Default",
        learning_rate=0.1,
        step_result=TrainStepResult(
            task_key="Default",
            step=display_step - 1,
            train_results={"rmse": 1.0},
        ),
        rank_context=ctx,
    )


def test_resolve_chrome_trace_path_single_process() -> None:
    assert (
        resolve_chrome_trace_path("timeline.json", rank=0, world_size=1)
        == "timeline.json"
    )
    assert (
        resolve_chrome_trace_path("out/trace.json", rank=3, world_size=1)
        == "out/trace.json"
    )


def test_resolve_chrome_trace_path_distributed_rank_suffix() -> None:
    assert (
        resolve_chrome_trace_path("timeline.json", rank=0, world_size=4)
        == "timeline.rank0.json"
    )
    assert (
        resolve_chrome_trace_path("dir/chrome.json", rank=2, world_size=4)
        == "dir/chrome.rank2.json"
    )


def test_disabled_factory_returns_none_without_profiler(tmp_path: Path) -> None:
    observer = create_profiler_observer(
        {
            "enable_profiler": False,
            "profiling": False,
            "profiling_file": str(tmp_path / "timeline.json"),
            "tensorboard_log_dir": str(tmp_path / "log"),
        },
        rank_context=RankContext(),
    )
    assert observer is None
    assert list(tmp_path.iterdir()) == []


def test_profiling_only_exports_chrome_and_steps_once(tmp_path: Path) -> None:
    chrome = tmp_path / "custom_timeline.json"
    profiler = MagicMock()
    schedule = MagicMock(return_value="sched")
    handler = MagicMock()
    factory = MagicMock(return_value=profiler)

    observer = TorchProfilerObserver(
        enable_profiler=False,
        profiling=True,
        profiling_file=str(chrome),
        tensorboard_log_dir=str(tmp_path / "log"),
        rank_context=RankContext(),
        profiler_factory=factory,
        schedule_factory=schedule,
        tensorboard_handler_factory=handler,
    )
    assert observer.wants_step(1)
    assert observer.wants_step(17)

    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    schedule.assert_called_once_with(wait=1, warmup=15, active=3, repeat=1)
    factory.assert_called_once()
    kwargs = factory.call_args.kwargs
    assert kwargs["on_trace_ready"] is None
    assert kwargs["schedule"] == "sched"
    profiler.start.assert_called_once()
    handler.assert_not_called()
    assert not (tmp_path / "log").exists()

    for display_step in (1, 2, 3):
        observer.on_step_end(_step(display_step))
    assert profiler.step.call_count == 3

    observer.on_train_end(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    profiler.stop.assert_called_once()
    profiler.export_chrome_trace.assert_called_once_with(str(chrome))
    assert observer.profiler is None


def test_enable_profiler_only_uses_tensorboard_handler(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    profiler = MagicMock()
    schedule = MagicMock(return_value="sched")
    ready = object()
    handler = MagicMock(return_value=ready)
    factory = MagicMock(return_value=profiler)

    observer = TorchProfilerObserver(
        enable_profiler=True,
        profiling=False,
        profiling_file=str(tmp_path / "timeline.json"),
        tensorboard_log_dir=str(log_dir),
        rank_context=RankContext(),
        profiler_factory=factory,
        schedule_factory=schedule,
        tensorboard_handler_factory=handler,
    )
    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    assert log_dir.is_dir()
    handler.assert_called_once_with(str(log_dir), worker_name=None)
    assert factory.call_args.kwargs["on_trace_ready"] is ready

    observer.on_step_end(_step(1))
    observer.close()
    profiler.stop.assert_called_once()
    profiler.export_chrome_trace.assert_not_called()


def test_combined_modes_one_session_both_outputs(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    chrome = tmp_path / "both.json"
    profiler = MagicMock()
    schedule = MagicMock(return_value="sched")
    ready = object()
    handler = MagicMock(return_value=ready)
    factory = MagicMock(return_value=profiler)

    observer = create_profiler_observer(
        {
            "enable_profiler": True,
            "profiling": True,
            "profiling_file": str(chrome),
            "tensorboard_log_dir": str(log_dir),
        },
        rank_context=RankContext(),
    )
    assert observer is not None
    observer._profiler_factory = factory
    observer._schedule_factory = schedule
    observer._tensorboard_handler_factory = handler

    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    assert factory.call_count == 1
    assert factory.call_args.kwargs["on_trace_ready"] is ready
    observer.on_step_end(_step(1))
    observer.on_step_end(_step(2))
    observer.close()

    assert profiler.step.call_count == 2
    profiler.export_chrome_trace.assert_called_once_with(str(chrome))
    profiler.stop.assert_called_once()


def test_distributed_rank_suffix_and_worker_name(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    chrome = tmp_path / "timeline.json"
    profiler = MagicMock()
    schedule = MagicMock(return_value="sched")
    handler = MagicMock(return_value=object())
    factory = MagicMock(return_value=profiler)
    ctx = RankContext(rank=1, world_size=4)

    observer = TorchProfilerObserver(
        enable_profiler=True,
        profiling=True,
        profiling_file=str(chrome),
        tensorboard_log_dir=str(log_dir),
        rank_context=ctx,
        profiler_factory=factory,
        schedule_factory=schedule,
        tensorboard_handler_factory=handler,
    )
    assert observer.chrome_trace_path == str(tmp_path / "timeline.rank1.json")

    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=ctx,
    )
    handler.assert_called_once_with(str(log_dir), worker_name="rank1")
    observer.on_step_end(_step(1, rank_context=ctx))
    observer.close()
    profiler.export_chrome_trace.assert_called_once_with(
        str(tmp_path / "timeline.rank1.json")
    )


def test_exceptional_exit_still_releases(tmp_path: Path) -> None:
    profiler = MagicMock()
    observer = TorchProfilerObserver(
        enable_profiler=False,
        profiling=True,
        profiling_file=str(tmp_path / "timeline.json"),
        tensorboard_log_dir=str(tmp_path / "log"),
        rank_context=RankContext(),
        profiler_factory=MagicMock(return_value=profiler),
        schedule_factory=MagicMock(return_value="sched"),
        tensorboard_handler_factory=MagicMock(),
    )
    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    # Simulate exceptional exit via the same finally path AbstractTrainer uses.
    observer.on_train_end(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    profiler.stop.assert_called_once()
    assert observer.profiler is None
    # Second close is a no-op.
    observer.close()
    assert profiler.stop.call_count == 1


def test_close_releases_even_when_export_fails(tmp_path: Path) -> None:
    profiler = MagicMock()
    profiler.export_chrome_trace.side_effect = RuntimeError("disk full")
    observer = TorchProfilerObserver(
        enable_profiler=False,
        profiling=True,
        profiling_file=str(tmp_path / "timeline.json"),
        tensorboard_log_dir=str(tmp_path / "log"),
        rank_context=RankContext(),
        profiler_factory=MagicMock(return_value=profiler),
        schedule_factory=MagicMock(return_value="sched"),
        tensorboard_handler_factory=MagicMock(),
    )
    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    try:
        observer.close()
    except RuntimeError:
        pass
    assert observer.profiler is None
    profiler.stop.assert_called_once()
