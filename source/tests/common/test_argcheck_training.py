# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for normalized training arguments."""

import pytest
from dargs.dargs import (
    ArgumentTypeError,
)

from deepmd.utils.argcheck import (
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
