# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

from deepmd.dpmodel.train import (
    AbstractTrainer,
    CheckpointObservation,
    DisplayObservation,
    RankContext,
    StepObservation,
    TrainerConfig,
    TrainingObserver,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
)


class RecordingObserver(TrainingObserver):
    def __init__(self, *, freq: int = 1) -> None:
        self.freq = freq
        self.events: list[str] = []

    def wants_step(self, display_step: int) -> bool:
        return display_step == 1 or (self.freq > 0 and display_step % self.freq == 0)

    def on_train_begin(self, tasks, *, rank_context: RankContext) -> None:
        self.events.append(f"begin:{rank_context.rank}")

    def on_step_end(self, observation: StepObservation) -> None:
        self.events.append(f"step:{observation.display_step}:{observation.task_key}")

    def on_display(self, observation: DisplayObservation) -> None:
        self.events.append(
            f"display:{observation.display_step}:{observation.timing.steps}"
        )

    def on_checkpoint(self, observation: CheckpointObservation) -> None:
        self.events.append(f"ckpt:{observation.display_step}")

    def on_train_end(self, tasks, *, rank_context: RankContext) -> None:
        self.events.append(f"end:{rank_context.rank}")


class DummyData:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def get_batch(self) -> dict[str, float]:
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return {"value": value}


class DummyTrainer(AbstractTrainer):
    def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
        batch = task.training_data.get_batch()
        return TrainStepResult(
            task_key=task.key,
            step=step,
            payload=batch,
            train_results={"rmse": batch["value"]},
        )

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
        return {"rmse": float(task.validation_data.get_batch()["value"])}

    def learning_rate(self, step: int) -> float:
        return 0.1 / (step + 1)

    def save_checkpoint(self, step: int) -> None:
        return None


def test_observer_hooks_follow_independent_frequencies(tmp_path: Path) -> None:
    observer = RecordingObserver(freq=2)
    trainer = DummyTrainer(
        TrainerConfig(
            num_steps=4,
            disp_file=str(tmp_path / "lcurve.out"),
            disp_freq=4,
            save_freq=4,
            timing_in_training=True,
        ),
        observers=observer,
    )
    trainer.run(
        TrainingTaskCollection.single(DummyData([1.0, 2.0, 3.0, 4.0]), DummyData([9.0]))
    )

    assert observer.events[0] == "begin:0"
    assert "step:1:Default" in observer.events
    assert "step:2:Default" in observer.events
    assert "step:3:Default" not in observer.events
    assert "step:4:Default" in observer.events
    # Display still follows disp_freq (plus the opening step); TB step hooks
    # follow tensorboard-style frequency independently.
    assert "display:1:1" in observer.events
    assert "display:4:3" in observer.events
    assert "ckpt:4" in observer.events
    assert observer.events[-1] == "end:0"


def test_observer_still_closes_after_training_failure(tmp_path: Path) -> None:
    observer = RecordingObserver(freq=1)

    class FailingTrainer(DummyTrainer):
        def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
            raise RuntimeError("boom")

    trainer = FailingTrainer(
        TrainerConfig(
            num_steps=1,
            disp_file=str(tmp_path / "lcurve.out"),
            disp_freq=1,
            save_freq=1,
        ),
        observers=observer,
    )
    try:
        trainer.run(TrainingTaskCollection.single(DummyData([1.0])))
    except RuntimeError:
        pass
    assert observer.events[0] == "begin:0"
    assert observer.events[-1] == "end:0"


def test_step_observation_is_skipped_when_no_observer_wants_it(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    class QuietObserver(TrainingObserver):
        def wants_step(self, display_step: int) -> bool:
            return False

        def on_step_end(self, observation: StepObservation) -> None:
            calls.append(observation.display_step)

    trainer = DummyTrainer(
        TrainerConfig(
            num_steps=3,
            disp_file=str(tmp_path / "lcurve.out"),
            disp_freq=3,
            save_freq=3,
            display_in_training=False,
        ),
        observers=QuietObserver(),
    )
    trainer.run(TrainingTaskCollection.single(DummyData([1.0, 2.0, 3.0])))
    assert calls == []
