# SPDX-License-Identifier: LGPL-3.0-or-later
"""TensorFlow 2 backend serialization contracts."""

import os

import pytest

if os.environ.get("DP_TEST_TF2_ONLY") != "1":
    pytest.skip(
        "TF2 tests require DP_TEST_TF2_ONLY=1",
        allow_module_level=True,
    )

from deepmd.tf2.utils import (
    serialization,
)


def test_checkpoint_serialization_declares_dense_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TF2 training checkpoint exposes its dense source semantics."""
    state = {
        "backend": "TensorFlow2",
        "model_def_script": {},
        "current_step": 0,
    }
    monkeypatch.setattr(
        serialization,
        "_load_checkpoint_state",
        lambda _path: ("checkpoint", state),
    )
    monkeypatch.setattr(
        serialization,
        "_restore_models_from_checkpoint",
        lambda _checkpoint, _script, _state: {},
    )
    monkeypatch.setattr(
        serialization,
        "_serialize_models",
        lambda _models, _script: {},
    )

    data = serialization.serialize_from_file("model.tf2")

    assert data["lower_input_kind"] == "nlist"
