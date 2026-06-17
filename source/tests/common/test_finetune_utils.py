# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

from deepmd.utils import finetune


def test_descriptor_config_warning_reports_nested_difference(monkeypatch, caplog):
    monkeypatch.setattr(
        finetune,
        "_normalize_descriptor_for_compare",
        lambda descriptor: descriptor,
    )

    input_descriptor = {
        "type": "dpa3",
        "repflow": {"nlayer": 6, "rcut": 6.0},
        "trainable": False,
    }
    pretrained_descriptor = {
        "type": "dpa3",
        "repflow": {"nlayer": 16, "rcut": 6.0},
        "trainable": True,
    }

    with caplog.at_level(logging.WARNING):
        finetune.warn_configuration_mismatch_during_finetune(
            input_descriptor,
            pretrained_descriptor,
        )

    assert "repflow.nlayer" in caplog.text
    assert "input=6, pretrained=16" in caplog.text
    assert "trainable" not in caplog.text


def test_descriptor_config_warning_skips_default_only_difference(monkeypatch, caplog):
    def normalize_with_default(descriptor):
        return {
            "activation_function": "tanh",
            **descriptor,
        }

    monkeypatch.setattr(
        finetune,
        "_normalize_descriptor_for_compare",
        normalize_with_default,
    )

    with caplog.at_level(logging.WARNING):
        finetune.warn_descriptor_config_differences(
            {"type": "se_e2_a"},
            {"type": "se_e2_a", "activation_function": "tanh"},
        )

    assert caplog.text == ""
