# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from copy import (
    deepcopy,
)

from deepmd.utils import (
    finetune,
)


def _model_config(
    type_map: list[str],
    *,
    descriptor_sel: list[int] | None = None,
    fitting_neuron: list[int] | None = None,
    trainable: bool = True,
) -> dict:
    return {
        "type_map": type_map,
        "descriptor": {
            "type": "se_e2_a",
            "sel": descriptor_sel or [1 for _ in type_map],
            "trainable": trainable,
        },
        "fitting_net": {
            "neuron": fitting_neuron or [4],
            "trainable": trainable,
        },
    }


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


def test_finetune_rule_builder_updates_single_task_config_preserving_trainable():
    pretrained = _model_config(["O", "H"], descriptor_sel=[8, 16], fitting_neuron=[32])
    target = _model_config(
        ["O", "H", "B"],
        descriptor_sel=[1, 1, 1],
        fitting_neuron=[2],
        trainable=False,
    )

    updated, links = finetune.FinetuneRuleBuilder(
        pretrained,
        target,
        change_model_params=True,
    ).build()

    rule = links["Default"]
    assert updated["descriptor"]["sel"] == [8, 16]
    assert updated["descriptor"]["trainable"] is False
    assert updated["fitting_net"]["neuron"] == [32]
    assert updated["fitting_net"]["trainable"] is False
    assert rule.get_finetune_tmap() == ["O", "H", "B"]
    assert rule.get_pretrained_tmap() == ["O", "H"]
    assert rule.get_has_new_type()


def test_finetune_rule_builder_random_fitting_keeps_target_fitting_net():
    pretrained = _model_config(["O", "H"], descriptor_sel=[8, 16], fitting_neuron=[32])
    target = _model_config(["O", "H"], descriptor_sel=[1, 1], fitting_neuron=[2])

    updated, links = finetune.FinetuneRuleBuilder(
        pretrained,
        target,
        model_branch="RANDOM",
        change_model_params=True,
    ).build()

    assert updated["descriptor"]["sel"] == [8, 16]
    assert updated["fitting_net"]["neuron"] == [2]
    assert links["Default"].get_random_fitting()


def test_finetune_rule_builder_rejects_unknown_branch_from_single_task():
    try:
        finetune.FinetuneRuleBuilder(
            _model_config(["O", "H"]),
            _model_config(["O", "H"], descriptor_sel=[1, 1]),
            model_branch="typo",
        ).build()
    except ValueError as exc:
        assert "Single-task pretrained models" in str(exc)
        assert "typo" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_finetune_rule_builder_handles_multitask_resume_branch_and_random():
    pretrained = {
        "model_dict": {
            "task_a": _model_config(["O", "H"], descriptor_sel=[8, 16]),
            "task_b": _model_config(["O", "H"], descriptor_sel=[4, 4]),
        }
    }
    target = {
        "model_dict": {
            "task_a": _model_config(["O", "H"], descriptor_sel=[1, 1]),
            "task_c": {
                **_model_config(["O", "H"], descriptor_sel=[1, 1]),
                "finetune_head": "task_b",
            },
            "task_d": _model_config(
                ["O", "H"], descriptor_sel=[1, 1], fitting_neuron=[7]
            ),
        }
    }

    updated, links = finetune.FinetuneRuleBuilder(
        pretrained,
        target,
        change_model_params=True,
    ).build()

    assert links["task_a"].get_resuming()
    assert links["task_a"].get_model_branch() == "task_a"
    assert not links["task_c"].get_resuming()
    assert links["task_c"].get_model_branch() == "task_b"
    assert not links["task_c"].get_random_fitting()
    assert links["task_d"].get_model_branch() == "task_a"
    assert links["task_d"].get_random_fitting()
    assert updated["model_dict"]["task_c"]["descriptor"]["sel"] == [4, 4]
    assert updated["model_dict"]["task_d"]["descriptor"]["sel"] == [8, 16]
    assert updated["model_dict"]["task_d"]["fitting_net"]["neuron"] == [7]


def test_finetune_rule_builder_accepts_multitask_finetune_head_alias():
    pretrained = {
        "model_dict": {
            "task_a": _model_config(["O", "H"], descriptor_sel=[8, 16]),
            "task_b": {
                **_model_config(["O", "H"], descriptor_sel=[4, 4]),
                "model_branch_alias": ["alias_b"],
            },
        }
    }
    target = {
        "model_dict": {
            "task_c": {
                **_model_config(["O", "H"], descriptor_sel=[1, 1]),
                "finetune_head": "alias_b",
            },
        }
    }

    updated, links = finetune.FinetuneRuleBuilder(
        pretrained,
        target,
        change_model_params=True,
    ).build()

    assert links["task_c"].get_model_branch() == "task_b"
    assert updated["model_dict"]["task_c"]["descriptor"]["sel"] == [4, 4]


def test_finetune_rule_builder_does_not_mutate_input_config():
    target = _model_config(["O", "H"], descriptor_sel=[1, 1], fitting_neuron=[2])
    target_before = deepcopy(target)

    finetune.FinetuneRuleBuilder(
        _model_config(["O", "H"], descriptor_sel=[8, 16], fitting_neuron=[32]),
        target,
        change_model_params=True,
    ).build()

    assert target == target_before


def test_finetune_rule_builder_rejects_multitask_cli_branch():
    pretrained = {"model_dict": {"task_a": _model_config(["O", "H"])}}
    target = {"model_dict": {"task_a": _model_config(["O", "H"])}}

    try:
        finetune.FinetuneRuleBuilder(
            pretrained,
            target,
            model_branch="task_a",
        ).build()
    except ValueError as exc:
        assert "Multi-task fine-tuning" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_finetune_rule_builder_single_task_target_auto_picks_multitask_branch():
    """A single-task target (e.g. GroupPropertyModel) finetuning directly off a
    multi-task foundation-model checkpoint, with no CLI ``--model-branch`` and
    no ``finetune_head`` set, must auto-select the checkpoint's only branch and
    randomly re-initialize the fitting net rather than erroring out.
    """
    pretrained = {
        "model_dict": {
            "Alex2D": _model_config(["O", "H"], descriptor_sel=[8, 16]),
        }
    }
    target = _model_config(["O", "H"], descriptor_sel=[1, 1], fitting_neuron=[2])

    updated, links = finetune.FinetuneRuleBuilder(
        pretrained,
        target,
        change_model_params=True,
    ).build()

    rule = links["Default"]
    assert rule.get_model_branch() == "Alex2D"
    assert rule.get_random_fitting()
    assert updated["descriptor"]["sel"] == [8, 16]
    # fitting net stays the target's own (freshly-initialized) spec
    assert updated["fitting_net"]["neuron"] == [2]


def test_finetune_rule_builder_single_task_target_picks_first_of_several_branches():
    """Documents current (non-error) behavior: with several branches and no
    explicit selection, the builder deterministically takes the first one in
    insertion order rather than raising an ambiguity error.
    """
    pretrained = {
        "model_dict": {
            "first": _model_config(["O", "H"], descriptor_sel=[8, 16]),
            "second": _model_config(["O", "H"], descriptor_sel=[4, 4]),
        }
    }
    target = _model_config(["O", "H"], descriptor_sel=[1, 1])

    _, links = finetune.FinetuneRuleBuilder(
        pretrained,
        target,
        change_model_params=True,
    ).build()

    assert links["Default"].get_model_branch() == "first"
