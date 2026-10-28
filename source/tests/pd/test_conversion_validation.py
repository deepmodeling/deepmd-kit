# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unsupported conversion must fail before Paddle reads training data."""

from copy import (
    deepcopy,
)

import pytest

from deepmd.pd.entrypoints import main as entrypoint


@pytest.mark.parametrize("multi_task", [False, True])
@pytest.mark.parametrize("converted_key", ["training_data", "validation_data"])
def test_lmdb_conversion_fails_before_neighbor_stat(
    monkeypatch, multi_task, converted_key
):
    """Validate every task, including validation-only LMDB, before conversion."""
    native = {"systems": "native", "batch_size": 1}
    task = {"training_data": native, "validation_data": native}
    task[converted_key] = [{"systems": "input.extxyz", "format": "extxyz"}]
    model = {"type_map": ["H"]}
    config = {"model": model, "training": task}
    if multi_task:
        config = {
            "model": {"model_dict": {"a": model, "b": model}},
            "training": {
                "data_dict": {
                    "a": {"training_data": native},
                    "b": task,
                }
            },
        }
    monkeypatch.setattr(entrypoint, "SummaryPrinter", lambda: lambda: None)
    monkeypatch.setattr(entrypoint, "j_loader", lambda _: deepcopy(config))
    monkeypatch.setattr(entrypoint, "expand_model_preset", lambda value: value)
    monkeypatch.setattr(
        entrypoint, "preprocess_shared_params", lambda value: (value, None)
    )
    monkeypatch.setattr(entrypoint, "update_deepmd_input", lambda value, **_: value)
    monkeypatch.setattr(entrypoint, "normalize", lambda value, **_: value)

    def unexpected_data_access(*args, **kwargs):
        raise AssertionError("Data access started before LMDB rejection")

    monkeypatch.setattr(entrypoint, "get_data", unexpected_data_access)
    monkeypatch.setattr(entrypoint, "get_trainer", unexpected_data_access)
    with pytest.raises(
        NotImplementedError, match="Paddle backend does not support LMDB"
    ):
        entrypoint.train("input.json", None, None, None, None, "RANDOM")
