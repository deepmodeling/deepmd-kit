# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for normalized training arguments."""

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
