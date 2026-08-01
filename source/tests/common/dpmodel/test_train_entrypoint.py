# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import pytest

from deepmd.dpmodel.train import (
    AbstractTrainEntrypoint,
    TrainEntrypointOptions,
)


class RecordingTrainEntrypoint(AbstractTrainEntrypoint):
    def __init__(
        self,
        *,
        fail_setup: bool = False,
        fail_training: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.fail_setup = fail_setup
        self.fail_training = fail_training
        self.neighbor_stat: Any = "unset"

    def prepare_options(
        self,
        options: TrainEntrypointOptions,
    ) -> TrainEntrypointOptions:
        self.events.append("prepare_options")
        return options

    def load_config(self, input_file: str) -> dict[str, Any]:
        self.events.append(f"load_config:{input_file}")
        return {
            "model": {},
            "training": {},
        }

    def validate_options(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> None:
        self.events.append("validate_options")

    def preprocess_config(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> dict[str, Any]:
        self.events.append("preprocess_config")
        config["model"]["model_dict"] = {"task": {"type_map": ["O"]}}
        return config

    def update_input(self, config: dict[str, Any]) -> dict[str, Any]:
        self.events.append("update_input")
        config["compat_updated"] = True
        return config

    def normalize_config(
        self,
        config: dict[str, Any],
        *,
        multi_task: bool,
    ) -> dict[str, Any]:
        self.events.append(f"normalize_config:{multi_task}")
        config["normalized"] = True
        return config

    def update_neighbor_stat(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        *,
        multi_task: bool,
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        self.events.append(f"update_neighbor_stat:{multi_task}")
        config["neighbor_updated"] = True
        return config, {"multi_task": multi_task}

    def dump_config(self, config: dict[str, Any], output: str) -> None:
        self.events.append(f"dump_config:{Path(output).name}")
        super().dump_config(config, output)

    def print_summary(self) -> None:
        self.events.append("print_summary")

    def setup_run(
        self,
        options: TrainEntrypointOptions,
        config: dict[str, Any],
    ) -> None:
        self.events.append("setup_run")
        if self.fail_setup:
            raise RuntimeError("setup failed")

    def teardown_run(
        self,
        options: TrainEntrypointOptions,
        config: dict[str, Any],
    ) -> None:
        self.events.append("teardown_run")

    def run_training(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        neighbor_stat: Any,
    ) -> None:
        self.events.append("run_training")
        self.neighbor_stat = neighbor_stat
        if self.fail_training:
            raise RuntimeError("training failed")


def test_train_entrypoint_runs_common_pipeline(tmp_path: Path) -> None:
    entrypoint = RecordingTrainEntrypoint()
    output = tmp_path / "out.json"

    entrypoint.run(
        TrainEntrypointOptions(
            input_file="input.json",
            output=str(output),
        )
    )

    assert entrypoint.events == [
        "prepare_options",
        "load_config:input.json",
        "validate_options",
        "preprocess_config",
        "update_input",
        "normalize_config:True",
        "update_neighbor_stat:True",
        "dump_config:out.json",
        "print_summary",
        "setup_run",
        "run_training",
        "teardown_run",
    ]
    assert entrypoint.neighbor_stat == {"multi_task": True}
    assert '"normalized": true' in output.read_text()


def test_train_entrypoint_can_skip_neighbor_stat(tmp_path: Path) -> None:
    entrypoint = RecordingTrainEntrypoint()

    entrypoint.run(
        TrainEntrypointOptions(
            input_file="input.json",
            output=str(tmp_path / "out.json"),
            skip_neighbor_stat=True,
        )
    )

    assert "update_neighbor_stat:True" not in entrypoint.events
    assert entrypoint.neighbor_stat is None


def test_train_entrypoint_tears_down_after_training_error(tmp_path: Path) -> None:
    entrypoint = RecordingTrainEntrypoint(fail_training=True)

    with pytest.raises(RuntimeError, match="training failed"):
        entrypoint.run(
            TrainEntrypointOptions(
                input_file="input.json",
                output=str(tmp_path / "out.json"),
            )
        )

    assert entrypoint.events[-2:] == ["run_training", "teardown_run"]


def test_train_entrypoint_tears_down_after_setup_error(tmp_path: Path) -> None:
    entrypoint = RecordingTrainEntrypoint(fail_setup=True)

    with pytest.raises(RuntimeError, match="setup failed"):
        entrypoint.run(
            TrainEntrypointOptions(
                input_file="input.json",
                output=str(tmp_path / "out.json"),
            )
        )

    assert entrypoint.events[-2:] == ["setup_run", "teardown_run"]
