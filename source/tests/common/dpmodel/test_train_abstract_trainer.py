# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import pytest

from deepmd.dpmodel.train import (
    AbstractTrainer,
    LearningCurveWriter,
    RankContext,
    TrainerConfig,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
)


class DummyData:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0
        self.requirements: list[Any] = []

    def __len__(self) -> int:
        return len(self.values)

    def get_batch(self) -> dict[str, float]:
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return {"value": value}

    def add_data_requirements(self, requirements: list[Any]) -> None:
        self.requirements.extend(requirements)


class DummyTrainer(AbstractTrainer):
    def __init__(
        self,
        trainer_config: TrainerConfig,
        *,
        rank_context: RankContext | None = None,
    ) -> None:
        super().__init__(trainer_config, rank_context=rank_context)
        self.steps: list[tuple[str, int, float]] = []
        self.checkpoints: list[int] = []

    def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
        batch = task.training_data.get_batch()
        self.steps.append((task.key, step, batch["value"]))
        return TrainStepResult(task_key=task.key, step=step, payload=batch)

    def evaluate_training(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float]:
        if step_result is None or step_result.task_key != task.key:
            return {"rmse": 0.0}
        return {"rmse": float(step_result.payload["value"])}

    def evaluate_validation(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float] | None:
        if task.validation_data is None:
            return None
        values = [
            float(task.validation_data.get_batch()["value"])
            for _ in range(task.valid_numb_batch)
        ]
        return {"rmse": sum(values) / len(values)}

    def learning_rate(self, step: int) -> float:
        return 0.1 / (step + 1)

    def save_checkpoint(self, step: int) -> None:
        self.checkpoints.append(step)


def _lcurve_steps(path: Path) -> list[int]:
    steps = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        steps.append(int(line.split()[0]))
    return steps


def test_single_task_collection_adds_requirements() -> None:
    train_data = DummyData([1.0])
    valid_data = DummyData([2.0])
    tasks = TrainingTaskCollection.single(
        train_data,
        valid_data,
        data_requirements=["energy"],
    )

    task = tasks.select()
    task.add_data_requirements()

    assert not tasks.is_multitask
    assert tasks.select() is task
    assert train_data.requirements == ["energy"]
    assert valid_data.requirements == ["energy"]


def test_task_collection_rejects_duplicate_sequence_keys() -> None:
    with pytest.raises(ValueError, match="unique"):
        TrainingTaskCollection(
            [
                TrainingTask("task", DummyData([1.0])),
                TrainingTask("task", DummyData([2.0])),
            ]
        )


def test_learning_curve_row_uses_training_metric_order() -> None:
    row = LearningCurveWriter().format_row(
        step=1,
        learning_rate=0.1,
        train_results={"rmse": 1.0, "mae": 2.0},
        valid_results={"mae": 3.0},
    )

    assert row.split() == ["1", "nan", "1.00e+00", "3.00e+00", "2.00e+00", "1.0e-01"]


def test_abstract_trainer_drives_single_task_loop(tmp_path: Path) -> None:
    lcurve = tmp_path / "lcurve.out"
    trainer = DummyTrainer(
        TrainerConfig(
            num_steps=3,
            disp_file=str(lcurve),
            disp_freq=2,
            save_freq=2,
            timing_in_training=False,
        )
    )
    tasks = TrainingTaskCollection.single(
        DummyData([1.0, 2.0, 3.0]),
        DummyData([10.0, 20.0]),
        valid_numb_batch=2,
    )

    trainer.run(tasks)

    assert trainer.steps == [
        ("Default", 0, 1.0),
        ("Default", 1, 2.0),
        ("Default", 2, 3.0),
    ]
    assert trainer.checkpoints == [2, 3]
    assert _lcurve_steps(lcurve) == [1, 2]
    assert "rmse_val" in lcurve.read_text()


def test_non_chief_rank_skips_user_visible_outputs(tmp_path: Path) -> None:
    lcurve = tmp_path / "lcurve.out"
    trainer = DummyTrainer(
        TrainerConfig(
            num_steps=3,
            disp_file=str(lcurve),
            disp_freq=1,
            save_freq=1,
            timing_in_training=False,
        ),
        rank_context=RankContext(rank=1, world_size=2),
    )
    tasks = TrainingTaskCollection.single(
        DummyData([1.0, 2.0, 3.0]),
        DummyData([10.0]),
    )

    trainer.run(tasks)

    assert trainer.steps == [
        ("Default", 0, 1.0),
        ("Default", 1, 2.0),
        ("Default", 2, 3.0),
    ]
    assert trainer.checkpoints == []
    assert not lcurve.exists()


def test_abstract_trainer_tears_down_when_lcurve_open_fails(tmp_path: Path) -> None:
    events: list[str] = []

    def record_begin(tasks: TrainingTaskCollection) -> None:
        events.append("begin")

    def record_end(tasks: TrainingTaskCollection) -> None:
        events.append("end")

    trainer = DummyTrainer(
        TrainerConfig(
            num_steps=1,
            disp_file=str(tmp_path / "missing" / "lcurve.out"),
            disp_freq=1,
            save_freq=1,
        )
    )
    trainer.on_train_begin = record_begin
    trainer.on_train_end = record_end

    with pytest.raises(FileNotFoundError):
        trainer.run(
            TrainingTaskCollection.single(
                DummyData([1.0]),
                None,
            )
        )

    assert events == ["begin", "end"]


def test_multitask_training_uses_single_task_as_collection_item(
    tmp_path: Path,
) -> None:
    lcurve = tmp_path / "lcurve.out"
    trainer = DummyTrainer(
        TrainerConfig(
            num_steps=1,
            disp_file=str(lcurve),
            disp_freq=1,
            save_freq=1,
            timing_in_training=False,
        )
    )
    tasks = TrainingTaskCollection(
        [
            TrainingTask(
                "task_a",
                DummyData([1.0]),
                DummyData([11.0]),
            ),
            TrainingTask(
                "task_b",
                DummyData([2.0]),
                DummyData([22.0]),
            ),
        ],
        probabilities={"task_a": 1.0, "task_b": 0.0},
    )

    trainer.run(tasks)

    assert trainer.steps == [("task_a", 0, 1.0)]
    assert trainer.checkpoints == [1]
    assert tasks["task_b"].training_data.index == 0
    lcurve_text = lcurve.read_text()
    assert "rmse_val_task_a" in lcurve_text
    assert "rmse_trn_task_b" in lcurve_text
