# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

from deepmd.utils import (
    finetune,
)


def test_descriptor_normalization_uses_descriptor_type_count():
    assert finetune._infer_synthetic_type_count({"sel": [16, 24, 32]}) == 3
    assert finetune._infer_synthetic_type_count({"exclude_types": [[0, 3]]}) == 4


def test_descriptor_config_warning_reports_nested_difference(monkeypatch, caplog):
    monkeypatch.setattr(
        finetune,
        "_normalize_descriptor_for_compare",
        lambda descriptor: descriptor,
    )

    input_descriptor = {
        "type": "dpa3",
        "repflow": {"nlayers": 6, "rcut": 6.0},
        "trainable": False,
    }
    pretrained_descriptor = {
        "type": "dpa3",
        "repflow": {"nlayers": 16, "rcut": 6.0},
        "trainable": True,
    }

    with caplog.at_level(logging.WARNING):
        finetune.warn_configuration_mismatch_during_finetune(
            input_descriptor,
            pretrained_descriptor,
        )

    assert "repflow.nlayers" in caplog.text
    assert "input=6, pretrained=16" in caplog.text
    assert "trainable" not in caplog.text


def test_descriptor_config_warning_skips_default_only_difference(caplog):
    with caplog.at_level(logging.WARNING):
        finetune.warn_descriptor_config_differences(
            {"type": "se_e2_a", "sel": [16, 16], "rcut": 6.0},
            {
                "type": "se_e2_a",
                "sel": [16, 16],
                "rcut": 6.0,
                "activation_function": "tanh",
            },
        )

    assert caplog.text == ""


def test_descriptor_config_warning_falls_back_to_raw_if_normalization_fails(
    monkeypatch, caplog
):
    input_descriptor = {"type": "dpa3", "repflow": {"nlayers": 6}}
    pretrained_descriptor = {"type": "dpa3", "repflow": {"nlayers": 16}}

    def normalize_one_side_then_fail(descriptor):
        if descriptor is pretrained_descriptor:
            raise ValueError("legacy schema")
        return {**descriptor, "implicit_default": True}

    monkeypatch.setattr(
        finetune,
        "_normalize_descriptor_for_compare",
        normalize_one_side_then_fail,
    )

    with caplog.at_level(logging.WARNING):
        finetune.warn_configuration_mismatch_during_finetune(
            input_descriptor,
            pretrained_descriptor,
        )

    assert "repflow.nlayers" in caplog.text
    assert "implicit_default" not in caplog.text


def test_descriptor_config_warning_distinguishes_none_from_missing(monkeypatch, caplog):
    monkeypatch.setattr(
        finetune,
        "_normalize_descriptor_for_compare",
        lambda descriptor: descriptor,
    )

    with caplog.at_level(logging.WARNING):
        finetune.warn_configuration_mismatch_during_finetune(
            {"type": "dpa3", "input_none": None},
            {"type": "dpa3", "pretrained_none": None},
        )

    assert "input_none: input=None, pretrained=(missing)" in caplog.text
    assert "pretrained_none: input=(missing), pretrained=None" in caplog.text
