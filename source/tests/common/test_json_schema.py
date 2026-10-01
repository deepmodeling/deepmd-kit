# SPDX-License-Identifier: LGPL-3.0-or-later
"""JSON Schema validation of input files before preset expansion."""

import copy
import json
from typing import (
    Any,
)

import pytest
from jsonschema import (
    Draft202012Validator,
)

from deepmd.utils.argcheck import (
    gen_json_schema,
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.model_preset import (
    MODEL_PRESETS,
)


@pytest.fixture(scope="module")
def validators() -> dict[bool, Draft202012Validator]:
    result = {}
    for multi_task in (False, True):
        schema = json.loads(gen_json_schema(multi_task=multi_task))
        Draft202012Validator.check_schema(schema)
        result[multi_task] = Draft202012Validator(schema)
    return result


def _input(model: dict[str, Any], multi_task: bool = False) -> dict[str, Any]:
    training = {"training_data": {"systems": ["dummy"]}, "numb_steps": 10}
    if multi_task:
        training = {
            "data_dict": {"a": {"training_data": {"systems": ["dummy"]}}},
            "numb_steps": 10,
        }
    return {"model": model, "training": training}


def _assert_valid(validator: Draft202012Validator, data: dict[str, Any]) -> None:
    errors = list(validator.iter_errors(data))
    assert not errors, [(list(error.path), error.message) for error in errors]


def test_preset_user_example(validators: dict[bool, Draft202012Validator]) -> None:
    data = _input(
        {
            "preset": "dpa4-ultra-v20260911",
            "descriptor": {"use_amp": True, "seed": 42},
            "fitting_net": {"seed": 42},
            "use_compile": True,
            "enable_tf32": True,
            "_comment": "that's all",
        }
    )
    data["learning_rate"] = {
        "type": "cosine",
        "start_lr": 1.2e-4,
        "stop_lr": 1e-6,
        "warmup_ratio": 0.003,
        "warmup_start_factor": 0.2,
    }
    normalized = normalize(update_deepmd_input(copy.deepcopy(data), warning=False))
    assert normalized["model"]["descriptor"]["type"] == "dpa4"
    _assert_valid(validators[False], data)


@pytest.mark.parametrize("name", sorted(MODEL_PRESETS))
def test_every_preset_allows_partial_overrides(
    validators: dict[bool, Draft202012Validator], name: str
) -> None:
    for model in (
        {"preset": name},
        {
            "preset": name.upper(),
            "descriptor": {"use_amp": True, "seed": 42},
            "fitting_net": {"seed": 42},
        },
    ):
        _assert_valid(validators[False], _input(model))


@pytest.mark.parametrize(
    "overrides",
    [
        {"preset": "dpa4-unknown-v20260911"},
        {"preset": "dpa4-nano-v20260911\n"},
        {"preset": 42},
        {"descriptor": {"seed": "forty-two"}},
        {"descriptor": {"use_amp": "true"}},
        {"descriptor": {"type": "unknown"}},
        {"fitting_net": {"seed": "forty-two"}},
        {"fitting_net": {"neuron": "wide"}},
        {"fitting_net": {"type": "unknown"}},
        {
            "fitting_net": {
                "type": "property",
                "property_name": "charge",
                "task_dim": "four",
            }
        },
        {"use_compile": "true"},
        {"type": "unknown"},
        {"type": "frozen"},
        {"spin": {"scheme": "native"}},
    ],
)
def test_presets_preserve_field_validation(
    validators: dict[bool, Draft202012Validator], overrides: dict[str, Any]
) -> None:
    model = {"preset": "dpa4-nano-v20260911", **overrides}
    assert not validators[False].is_valid(_input(model))


def test_explicit_types_and_aliases(
    validators: dict[bool, Draft202012Validator],
) -> None:
    _assert_valid(
        validators[False],
        _input(
            {
                "preset": "dpa4-nano-v20260911",
                "type": "SeZM",
                "descriptor": {"type": "SeZM", "so2_layers": 3},
                "fitting_net": {
                    "type": "property",
                    "property_name": "charge",
                    "task_dim": 4,
                },
            }
        ),
    )


def test_plain_models_keep_required_fields(
    validators: dict[bool, Draft202012Validator],
) -> None:
    model = {
        "type_map": ["O"],
        "descriptor": {"type": "se_e2_a", "sel": [10]},
        "fitting_net": {"neuron": [4]},
    }
    _assert_valid(validators[False], _input(model))
    del model["descriptor"]["type"]
    assert not validators[False].is_valid(_input(model))


def test_multi_task_preset_inheritance(
    validators: dict[bool, Draft202012Validator],
) -> None:
    model = {
        "preset": "DPA4-Nano-v20260911",
        "descriptor": {"use_amp": True},
        "fitting_net": {"seed": 42},
        "model_dict": {
            "a": {},
            "b": {"preset": "dpa4c-mini-v20260911", "descriptor": {"seed": 42}},
        },
    }
    _assert_valid(validators[True], _input(model, multi_task=True))
    model["model_dict"]["b"]["descriptor"]["seed"] = "forty-two"
    assert not validators[True].is_valid(_input(model, multi_task=True))


def test_multi_task_branch_preset_and_shared_references(
    validators: dict[bool, Draft202012Validator],
) -> None:
    model = {
        "shared_dict": {
            "type_map": ["O", "H"],
            "descriptor": {"type": "dpa4", "use_amp": True},
        },
        "model_dict": {
            "a": {
                "preset": "dpa4-nano-v20260911",
                "type_map": "type_map",
                "descriptor": "descriptor:1",
                "fitting_net": {"seed": 42},
            }
        },
    }
    _assert_valid(validators[True], _input(model, multi_task=True))
    model["preset"] = "dpa4-mini-v20260911"
    _assert_valid(validators[True], _input(model, multi_task=True))
    model["model_dict"]["a"]["preset"] = "unknown"
    assert not validators[True].is_valid(_input(model, multi_task=True))
