# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for normalized training arguments."""

import copy
from typing import (
    Any,
)

import pytest
from dargs.dargs import (
    ArgumentError,
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
