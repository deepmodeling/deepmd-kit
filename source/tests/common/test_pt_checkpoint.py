# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for PyTorch checkpoint compatibility helpers."""

import logging

from deepmd.utils.pt_checkpoint import (
    apply_pt_checkpoint_defaults,
    normalize_pt_model_params,
)


def test_normalize_property_fitting_net_defaults_distinguish_types(caplog) -> None:
    model_params = {
        "type_map": ["H"],
        "fitting_net": {"type": "property", "property_name": "foo"},
    }

    with caplog.at_level(logging.WARNING, logger="deepmd.utils.pt_checkpoint"):
        result = normalize_pt_model_params(model_params)

    assert result is model_params
    assert model_params["fitting_net"]["distinguish_types"] is False
    assert "missing from a property fitting network" in caplog.text


def test_normalize_pt_model_params_handles_multitask_and_shared_models(caplog) -> None:
    shared_fitting = {"type": "property"}
    model_params = {
        "shared_dict": {"property_fitting": shared_fitting},
        "model_dict": {
            "head_a": {"fitting_net": "property_fitting:1"},
            "head_b": {"fitting_net": {"type": "property"}},
        },
        "models": [{"fitting_net": {"type": "property"}}],
    }

    with caplog.at_level(logging.WARNING, logger="deepmd.utils.pt_checkpoint"):
        normalize_pt_model_params(model_params)

    assert shared_fitting["distinguish_types"] is False
    assert (
        model_params["model_dict"]["head_b"]["fitting_net"]["distinguish_types"]
        is False
    )
    assert model_params["models"][0]["fitting_net"]["distinguish_types"] is False
    assert caplog.text.count("missing from a property fitting network") == 1


def test_normalize_pt_model_params_preserves_explicit_and_non_property_values(
    caplog,
) -> None:
    model_params = {
        "fitting_net": {"type": "property", "distinguish_types": True},
        "other_fitting": {"type": "ener"},
    }

    with caplog.at_level(logging.WARNING, logger="deepmd.utils.pt_checkpoint"):
        normalize_pt_model_params(model_params)

    assert model_params["fitting_net"]["distinguish_types"] is True
    assert "distinguish_types" not in model_params["other_fitting"]
    assert not caplog.records


def test_apply_pt_checkpoint_defaults_preserves_explicit_config_values(caplog) -> None:
    checkpoint_params = {
        "fitting_net": {"type": "property", "property_name": "foo"},
    }
    model_params = {
        "fitting_net": {"type": "property", "property_name": "foo"},
    }

    with caplog.at_level(logging.WARNING, logger="deepmd.utils.pt_checkpoint"):
        apply_pt_checkpoint_defaults(model_params, checkpoint_params)

    assert model_params["fitting_net"]["distinguish_types"] is False
    assert "model config and checkpoint" in caplog.text

    explicit_params = {
        "fitting_net": {
            "type": "property",
            "property_name": "foo",
            "distinguish_types": True,
        },
    }
    with caplog.at_level(logging.WARNING, logger="deepmd.utils.pt_checkpoint"):
        apply_pt_checkpoint_defaults(explicit_params, checkpoint_params)

    assert explicit_params["fitting_net"]["distinguish_types"] is True
