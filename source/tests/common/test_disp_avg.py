# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end display-window contracts shared by PT and PT-expt."""

import math
from contextlib import (
    nullcontext,
)
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)
from unittest.mock import (
    patch,
)

import pytest

from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

torch = pytest.importorskip("torch")


def _config(multi_task: bool, validation: bool, disp_avg: bool) -> dict[str, Any]:
    model = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4, 8],
            "rcut": 3.0,
            "rcut_smth": 0.5,
            "neuron": [4, 8],
            "axis_neuron": 4,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {"neuron": [8], "precision": "float64", "seed": 1},
        "data_stat_nbatch": 1,
    }
    data = {
        "training_data": {
            "systems": [str(Path(__file__).parents[1] / "pt/water/data/data_0")],
            "batch_size": 1,
        }
    }
    if validation:
        data["validation_data"] = {**data["training_data"], "numb_batch": 1}
    loss = {
        "type": "ener",
        "start_pref_e": 1.0,
        "limit_pref_e": 1.0,
        "start_pref_f": 1.0,
        "limit_pref_f": 1.0,
    }
    config = {
        "model": model,
        "learning_rate": {
            "type": "exp",
            "start_lr": 1e-4,
            "stop_lr": 1e-6,
            "decay_steps": 100,
        },
        "loss": loss,
        "training": {
            **data,
            "numb_steps": 6,
            "disp_freq": 3,
            "disp_avg": disp_avg,
            "save_freq": 6,
            "seed": 1,
        },
    }
    if multi_task:
        config["model"] = {"model_dict": {"a": deepcopy(model), "b": deepcopy(model)}}
        del config["loss"]
        config["loss_dict"] = {"a": loss, "b": {**loss, "loss_func": "mae"}}
        for key in data:
            del config["training"][key]
        config["training"]["data_dict"] = {"a": deepcopy(data), "b": deepcopy(data)}
        config["training"]["model_prob"] = {"a": 0.5, "b": 0.5}
    return normalize(update_deepmd_input(config, warning=False), multi_task=multi_task)


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize(
    "multi_task,disp_avg,validation",
    [
        (False, False, False),
        (False, True, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_display_matches_optimization_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    multi_task: bool,
    disp_avg: bool,
    validation: bool,
) -> None:
    """Unequal task counts, unseen tasks and window resets preserve batch order."""
    if backend == "pt":
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )
    else:
        from deepmd.pt_expt.entrypoints.main import (
            get_trainer,
        )

    monkeypatch.chdir(tmp_path)
    trainer = get_trainer(_config(multi_task, validation, disp_avg))
    forward = trainer.wrapper.forward
    observations: list[tuple[str, dict[str, float]]] = []

    def record_forward(*args: Any, **kwargs: Any) -> Any:
        result = forward(*args, **kwargs)
        if trainer.wrapper.training:
            observations.append(
                (
                    kwargs.get("task_key", "Default"),
                    {
                        key: float(value.detach())
                        if torch.is_tensor(value)
                        else float(value)
                        for key, value in result[2].items()
                        if "l2_" not in key
                    },
                )
            )
        return result

    task_order = ["a", "a", "a", "b", "b", "a"]
    sampling = nullcontext()
    if multi_task:
        if backend == "pt":
            sampling = patch(
                "deepmd.pt.train.training.dp_random.choice",
                side_effect=[0, 0, 0, 1, 1, 0],
            )
        else:
            sampling = patch.object(
                trainer,
                "select_task",
                side_effect=[trainer.training_tasks[key] for key in task_order],
            )
    with (
        sampling,
        patch.object(trainer.wrapper, "forward", side_effect=record_forward),
        patch.object(trainer, "get_data", wraps=trainer.get_data) as get_data,
    ):
        trainer.run()

    training_reads = [
        call.kwargs.get("task_key", "Default")
        for call in get_data.call_args_list
        if call.kwargs.get("is_train", True)
    ]
    expected_tasks = task_order if multi_task else ["Default"] * 6
    assert training_reads == expected_tasks
    assert [key for key, _ in observations] == expected_tasks
    lines = (tmp_path / "lcurve.out").read_text().splitlines()
    columns = lines[0].split()[1:]
    rows = [line.split() for line in lines if not line.startswith("#")]
    assert [int(row[0]) for row in rows] == [1, 3, 6]
    assert any("_val" in column for column in columns) is validation
    previous_step = 0
    for row in rows:
        assert len(row) == len(columns)
        step = int(row[0])
        window = (
            observations[previous_step:step]
            if disp_avg
            else observations[step - 1 : step]
        )
        for index, column in enumerate(columns):
            if "_trn" not in column:
                continue
            metric, suffix = column.split("_trn", 1)
            key = suffix[1:] if multi_task else "Default"
            values = [metrics[metric] for task, metrics in window if task == key]
            if values:
                assert float(row[index]) == pytest.approx(
                    sum(values) / len(values), rel=5e-3
                )
            else:
                assert math.isnan(float(row[index])), column
        previous_step = step


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
def test_restart_starts_a_fresh_metric_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """An unfinished pre-checkpoint window does not affect resumed averages."""
    if backend == "pt":
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )
    else:
        from deepmd.pt_expt.entrypoints.main import (
            get_trainer,
        )

    monkeypatch.chdir(tmp_path)
    config = _config(multi_task=True, validation=False, disp_avg=True)
    config["training"].update(numb_steps=2, save_freq=2)
    initial = get_trainer(deepcopy(config))
    if backend == "pt":
        sampling = patch("deepmd.pt.train.training.dp_random.choice", return_value=0)
    else:
        sampling = patch.object(
            initial, "select_task", return_value=initial.training_tasks["a"]
        )
    with sampling:
        initial.run()

    config["training"]["numb_steps"] = 3
    resumed = get_trainer(config, restart_model=str(tmp_path / "model.ckpt.pt"))
    assert resumed.start_step > 0
    forward = resumed.wrapper.forward
    observed: list[dict[str, float]] = []

    def record_forward(*args: Any, **kwargs: Any) -> Any:
        result = forward(*args, **kwargs)
        if resumed.wrapper.training:
            observed.append(
                {
                    key: float(value.detach())
                    if torch.is_tensor(value)
                    else float(value)
                    for key, value in result[2].items()
                    if "l2_" not in key
                }
            )
        return result

    if backend == "pt":
        sampling = patch("deepmd.pt.train.training.dp_random.choice", return_value=1)
    else:
        sampling = patch.object(
            resumed, "select_task", return_value=resumed.training_tasks["b"]
        )
    with sampling, patch.object(resumed.wrapper, "forward", side_effect=record_forward):
        resumed.run()
    assert len(observed) == config["training"]["numb_steps"] - resumed.start_step

    lines = (tmp_path / "lcurve.out").read_text().splitlines()
    columns = lines[0].split()[1:]
    rows = [line.split() for line in lines if not line.startswith("#")]
    assert [int(row[0]) for row in rows] == [1, 3]
    assert len(rows[-1]) == len(columns)
    for index, column in enumerate(columns):
        if "_trn" not in column:
            continue
        metric, suffix = column.split("_trn", 1)
        if suffix == "_a":
            assert math.isnan(float(rows[-1][index]))
        else:
            expected = sum(metrics[metric] for metrics in observed) / len(observed)
            assert float(rows[-1][index]) == pytest.approx(expected, rel=5e-3)
