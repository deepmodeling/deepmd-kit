# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for normalized training arguments."""

import copy
from typing import (
    Any,
)

import pytest
from dargs.dargs import (
    ArgumentError,
    ArgumentTypeError,
)

from deepmd.utils.argcheck import (
    normalize,
    training_args,
)


def test_ema_checkpoint_retention_is_left_for_runtime_inheritance() -> None:
    training_argument = training_args()

    normalized = training_argument.normalize_value(
        {"numb_steps": 1, "max_ckpt_keep": 7}
    )
    training_argument.check_value(normalized, strict=True)

    assert normalized["ema_ckpt_keep"] is None


@pytest.mark.parametrize("reset", [None, False, True])
def test_reset_data_progress_is_an_opt_in_boolean(reset) -> None:
    training_argument = training_args()
    values = {"numb_steps": 1}
    if reset is not None:
        values["reset_data_progress"] = reset

    normalized = training_argument.normalize_value(values)
    training_argument.check_value(normalized, strict=True)

    assert normalized["reset_data_progress"] is (False if reset is None else reset)


@pytest.mark.parametrize("reset", [None, 0, 1, "true", "false", [], {}])
def test_reset_data_progress_rejects_non_boolean_values(reset) -> None:
    training_argument = training_args()
    normalized = training_argument.normalize_value(
        {"numb_steps": 1, "reset_data_progress": reset}
    )

    with pytest.raises(ArgumentTypeError, match="reset_data_progress"):
        training_argument.check_value(normalized, strict=True)


@pytest.mark.parametrize("sampling", ["synchronized", "independent"])
def test_task_sampling_is_optional_and_multitask_only(sampling) -> None:
    argument = training_args(multi_task=True)
    values = {
        "numb_steps": 1,
        "data_dict": {
            key: {"training_data": {"systems": "dataset"}}
            for key in ("first", "second")
        },
    }
    normalized = argument.normalize_value(values)
    argument.check_value(normalized, strict=True)
    assert "task_sampling" not in normalized
    normalized = argument.normalize_value(values | {"task_sampling": sampling})
    argument.check_value(normalized, strict=True)
    assert normalized["task_sampling"] == sampling

    single = training_args()
    normalized = single.normalize_value({"numb_steps": 1, "task_sampling": sampling})
    with pytest.raises(ArgumentError, match="task_sampling"):
        single.check_value(normalized, strict=True)

    values["data_dict"].pop("second")
    normalized = argument.normalize_value(values | {"task_sampling": sampling})
    with pytest.raises(ValueError, match="at least two tasks"):
        argument.check_value(normalized, strict=True)


@pytest.mark.parametrize("sampling", ["uniform", "", None, 1])
def test_task_sampling_rejects_invalid_modes(sampling) -> None:
    argument = training_args(multi_task=True)
    normalized = argument.normalize_value(
        {
            "numb_steps": 1,
            "task_sampling": sampling,
            "data_dict": {
                key: {"training_data": {"systems": "dataset"}}
                for key in ("first", "second")
            },
        }
    )
    with pytest.raises(ArgumentError, match="task_sampling"):
        argument.check_value(normalized, strict=True)


def _training_input(data: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": {
            "type_map": ["H"],
            "descriptor": {"type": "se_e2_a", "sel": [1]},
            "fitting_net": {},
        },
        "training": {"numb_steps": 1, "training_data": data},
    }


def test_training_groups_normalize_aliases_without_mutating_input() -> None:
    source = _training_input(
        [
            {
                "systems": "first.lmdb",
                "prob": 0.6,
                "auto_prob_style": "prob_sys_size;0:1:1",
            },
            {"systems": ["second.lmdb", "third.lmdb"], "prob": 0.4},
        ]
    )
    original = copy.deepcopy(source)

    groups = normalize(source)["training"]["training_data"]

    assert source == original
    assert groups[0]["auto_prob"] == "prob_sys_size;0:1:1"
    assert "auto_prob_style" not in groups[0]
    assert groups[1]["auto_prob"] == "prob_sys_size"
    assert groups[0]["batch_size"] == groups[1]["batch_size"] == "auto"
    assert [group["prob"] for group in groups] == [0.6, 0.4]


def test_legacy_training_dictionary_remains_a_dictionary() -> None:
    group = normalize(_training_input({"systems": "dataset"}))["training"][
        "training_data"
    ]

    assert isinstance(group, dict)
    assert group["auto_prob"] == "prob_sys_size"
    assert group["batch_size"] == "auto"


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [{"systems": "dataset", "unknown_option": True}],
        [{"systems": "dataset", "prob": -0.1}],
        [{"systems": "dataset", "prob": 0.6}],
        [{"systems": "dataset", "prob": float("nan")}],
    ],
)
def test_invalid_training_groups_are_rejected(groups: list[dict[str, Any]]) -> None:
    with pytest.raises((ArgumentError, ValueError)):
        normalize(_training_input(groups))
