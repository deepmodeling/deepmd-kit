# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the TensorFlow 2 SavedModel inference adapter."""

import pytest

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
)
from deepmd.tf2.infer.deep_eval import (
    DeepEval,
)


def test_custom_neighbor_list_is_rejected_before_model_loading() -> None:
    """A custom ASE neighbor list must not be accepted as a silent no-op."""
    output_def = ModelOutputDef(FittingOutputDef([]))

    with pytest.raises(NotImplementedError, match="custom ASE neighbor_list"):
        DeepEval(
            "sentinel.savedmodeltf",
            output_def,
            neighbor_list=object(),
        )
