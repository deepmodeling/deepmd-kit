# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import datetime
from typing import (
    TYPE_CHECKING,
)
from unittest.mock import (
    MagicMock,
)

from deepmd.dpmodel.train.observer import (
    CheckpointObservation,
    DisplayObservation,
    StepObservation,
)
from deepmd.dpmodel.train.timing import (
    DisplayInterval,
)
from deepmd.dpmodel.train.trainer import (
    RankContext,
    TrainingTaskCollection,
    TrainStepResult,
)
from deepmd.pt_expt.train.tensorboard import (
    TensorBoardObserver,
    create_tensorboard_observer,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )


def _interval(display_step: int = 2) -> DisplayInterval:
    return DisplayInterval(
        display_step=display_step,
        wall_time=1.5,
        steps=2,
        eta=10,
        timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


def test_disabled_factory_returns_none_without_filesystem(tmp_path: Path) -> None:
    observer = create_tensorboard_observer(
        {"tensorboard": False, "tensorboard_log_dir": str(tmp_path / "log")},
        rank_context=RankContext(),
    )
    assert observer is None
    assert not (tmp_path / "log").exists()


def test_non_chief_never_creates_writer(tmp_path: Path) -> None:
    writer_factory = MagicMock()
    observer = TensorBoardObserver(
        log_dir=str(tmp_path / "log"),
        freq=1,
        rank_context=RankContext(rank=1, world_size=2),
        writer_factory=writer_factory,
    )
    assert not observer.enabled
    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(rank=1, world_size=2),
    )
    writer_factory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_chief_writes_at_configured_frequency_and_closes(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    writer = MagicMock()
    writer_factory = MagicMock(return_value=writer)
    observer = TensorBoardObserver(
        log_dir=str(log_dir),
        freq=2,
        rank_context=RankContext(),
        multi_task=False,
        writer_factory=writer_factory,
    )
    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    assert log_dir.is_dir()
    writer_factory.assert_called_once_with(log_dir=str(log_dir))

    for display_step, value in ((1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)):
        if not observer.wants_step(display_step):
            continue
        observer.on_step_end(
            StepObservation(
                step=display_step - 1,
                display_step=display_step,
                task_key="Default",
                learning_rate=0.01 * display_step,
                step_result=TrainStepResult(
                    task_key="Default",
                    step=display_step - 1,
                    train_results={"rmse": value},
                ),
                rank_context=RankContext(),
            )
        )

    observer.on_display(
        DisplayObservation(
            step=1,
            display_step=2,
            task_key="Default",
            learning_rate=0.02,
            train_results={"rmse": 2.0},
            valid_results={"rmse": 9.0},
            timing=_interval(2),
            rank_context=RankContext(),
        )
    )
    observer.on_checkpoint(
        CheckpointObservation(display_step=2, rank_context=RankContext())
    )
    observer.on_train_end(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )

    scalar_tags = [call.args[0] for call in writer.add_scalar.call_args_list]
    steps = [call.args[2] for call in writer.add_scalar.call_args_list]
    assert "learning_rate" in scalar_tags
    assert "train/rmse" in scalar_tags
    assert "valid/rmse" in scalar_tags
    assert "timing/interval_wall_time" in scalar_tags
    assert 1 in steps and 2 in steps and 4 in steps
    assert 3 not in steps
    writer.flush.assert_called()
    writer.close.assert_called_once()
    assert observer.writer is None


def test_restart_keeps_existing_event_directory(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    log_dir.mkdir()
    prior = log_dir / "events.out.tfevents.prior"
    prior.write_text("keep-me", encoding="utf-8")

    writer = MagicMock()
    writer_factory = MagicMock(return_value=writer)
    observer = TensorBoardObserver(
        log_dir=str(log_dir),
        freq=1,
        rank_context=RankContext(),
        writer_factory=writer_factory,
    )
    observer.on_train_begin(
        TrainingTaskCollection.single(object()),
        rank_context=RankContext(),
    )
    writer_factory.assert_called_once_with(log_dir=str(log_dir))
    assert prior.read_text(encoding="utf-8") == "keep-me"


def test_multi_task_tags_include_task_key(tmp_path: Path) -> None:
    writer = MagicMock()
    observer = TensorBoardObserver(
        log_dir=str(tmp_path / "tb"),
        freq=1,
        rank_context=RankContext(),
        multi_task=True,
        writer_factory=MagicMock(return_value=writer),
    )
    observer._writer = writer
    observer.on_step_end(
        StepObservation(
            step=0,
            display_step=1,
            task_key="water",
            learning_rate=0.1,
            step_result=TrainStepResult(
                task_key="water",
                step=0,
                train_results={"rmse": 1.25},
            ),
            rank_context=RankContext(),
        )
    )
    tags = [call.args[0] for call in writer.add_scalar.call_args_list]
    assert "train/water/rmse" in tags


def test_multi_process_only_chief_emits(tmp_path: Path) -> None:
    """Simulate two ranks: only the chief installs an active writer."""
    chiefs = []
    writers = []
    for rank in (0, 1):
        writer = MagicMock()
        writers.append(writer)
        observer = create_tensorboard_observer(
            {
                "tensorboard": True,
                "tensorboard_log_dir": str(tmp_path / f"rank{rank}"),
                "tensorboard_freq": 1,
            },
            rank_context=RankContext(rank=rank, world_size=2),
        )
        assert observer is not None
        observer._writer_factory = MagicMock(return_value=writer)
        observer.on_train_begin(
            TrainingTaskCollection.single(object()),
            rank_context=RankContext(rank=rank, world_size=2),
        )
        if observer.wants_step(1):
            observer.on_step_end(
                StepObservation(
                    step=0,
                    display_step=1,
                    task_key="Default",
                    learning_rate=0.1,
                    step_result=TrainStepResult(
                        task_key="Default",
                        step=0,
                        train_results={"rmse": 1.0},
                    ),
                    rank_context=RankContext(rank=rank, world_size=2),
                )
            )
        observer.close()
        chiefs.append(observer.enabled)

    assert chiefs == [True, False]
    assert writers[0].add_scalar.called
    assert not writers[1].add_scalar.called
    assert not writers[1].close.called


def test_exceptional_close_is_idempotent(tmp_path: Path) -> None:
    writer = MagicMock()
    observer = TensorBoardObserver(
        log_dir=str(tmp_path / "tb"),
        freq=1,
        rank_context=RankContext(),
        writer_factory=MagicMock(return_value=writer),
    )
    observer._writer = writer
    observer.close()
    observer.close()
    writer.close.assert_called_once()
