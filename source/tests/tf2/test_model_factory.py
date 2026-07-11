# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for TF2 DPA4/SeZM model-factory validation."""

import os

import pytest

if os.environ.get("DP_TEST_TF2_ONLY") != "1":
    pytest.skip(
        "TF2 tests require DP_TEST_TF2_ONLY=1",
        allow_module_level=True,
    )

from deepmd.tf2.model import model as model_module


def _base_sezm_config() -> dict:
    """Return the smallest config needed to exercise DPA4 factory routing."""
    return {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {"type": "dpa4"},
        "fitting_net": {"type": "dpa4_ener"},
    }


def test_null_blocks_receive_dpa4_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _base_sezm_config()
    data["descriptor"] = None
    data["fitting_net"] = None
    monkeypatch.setattr(model_module, "get_standard_model", lambda value: value)

    normalized = model_module.get_model(data)

    assert normalized["descriptor"]["type"] == "dpa4"
    assert normalized["fitting_net"]["type"] == "dpa4_ener"


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("spin", {}),
        ("bridging_method", "linear"),
        ("lora", {}),
        ("use_compile", True),
        ("preset_out_bias", [0.0]),
    ),
)
def test_rejects_unsupported_features(key: str, value: object) -> None:
    data = _base_sezm_config()
    data[key] = value

    with pytest.raises(NotImplementedError):
        model_module.get_model(data)


@pytest.mark.parametrize(
    ("section", "model_type"),
    (("descriptor", "se_e2_a"), ("fitting_net", "ener")),
)
def test_rejects_incompatible_component_types(
    section: str,
    model_type: str,
) -> None:
    data = _base_sezm_config()
    data[section]["type"] = model_type

    with pytest.raises(ValueError):
        model_module.get_model(data)


def test_rejects_mismatched_exclude_types() -> None:
    data = _base_sezm_config()
    data["descriptor"]["exclude_types"] = [[0, 1]]
    data["pair_exclude_types"] = [[1, 1]]

    with pytest.raises(ValueError):
        model_module.get_model(data)


def test_descriptor_exclude_types_feed_standard_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _base_sezm_config()
    data["descriptor"] = {"type": "SeZM", "exclude_types": [[0, 1]]}
    data["fitting_net"]["type"] = "sezm_ener"
    monkeypatch.setattr(model_module, "get_standard_model", lambda value: value)

    normalized = model_module.get_model(data)

    assert normalized["pair_exclude_types"] == [[0, 1]]
    assert normalized["descriptor"]["exclude_types"] == [[0, 1]]
