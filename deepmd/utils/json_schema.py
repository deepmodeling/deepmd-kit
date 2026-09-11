# SPDX-License-Identifier: LGPL-3.0-or-later
"""JSON Schema for model inputs before preset expansion."""

import json
import re
from typing import (
    Any,
)

from deepmd.utils.model_preset import (
    MODEL_PRESETS,
)

_REGIONS = ("type_map", "descriptor", "fitting_net")


def _preset_pattern(names: list[str]) -> str:
    """Match complete preset names case-insensitively with ECMA-262 syntax."""
    alternatives = [
        "".join(
            f"[{char.lower()}{char.upper()}]"
            if char.isalpha()
            else f"[{re.escape(char)}]"
            for char in name
        )
        for name in names
    ]
    # Unlike '$', the final assertion also excludes a trailing newline.
    return "^(?:" + "|".join(alternatives) + r")(?![\s\S])"


def _with_defaults(
    schema: dict[str, Any],
    defaults: dict[str, Any],
    ref: str,
    *,
    merge_regions: bool = True,
    multi_task: bool = False,
) -> dict[str, Any]:
    """Validate overrides while accounting for fields supplied by a preset.

    Model regions inherit missing fields; descriptor and fitting overrides
    merge one level only. Other nested objects retain their full contracts.
    Variant conditions use the inherited tag only when the input omits it.
    Unchanged field definitions are referenced to keep the schema compact.
    """
    result = {
        key: value
        for key, value in schema.items()
        if key not in ("properties", "required", "allOf")
    }
    properties = {}
    for name, field in schema.get("properties", {}).items():
        pointer = ref + "/properties/" + name.replace("~", "~0").replace("/", "~1")
        if merge_regions and name in ("descriptor", "fitting_net"):
            properties[name] = _with_defaults(
                field, defaults.get(name, {}), pointer, merge_regions=False
            )
        elif name == "type" and name in defaults:
            properties[name] = {**field, "default": defaults[name]}
        else:
            properties[name] = {"$ref": pointer}
        if multi_task and merge_regions and name in _REGIONS:
            properties[name] = {"anyOf": [properties[name], {"type": "string"}]}
    result["properties"] = properties
    result["required"] = [
        name for name in schema.get("required", []) if name not in defaults
    ]
    conditions = []
    for index, clause in enumerate(schema.get("allOf", [])):
        pointer = f"{ref}/allOf/{index}"
        if "if" in clause:
            condition = clause["if"]
            cases = condition["oneOf"]
            flag = next(iter(cases[0]["properties"]))
            if flag in defaults:
                tags = [case["properties"][flag]["const"] for case in cases]
                condition = {
                    "properties": {flag: {"enum": tags}},
                    "required": [] if defaults[flag] in tags else [flag],
                }
            conditions.append(
                {
                    "if": condition,
                    "then": _with_defaults(
                        clause["then"],
                        defaults,
                        pointer + "/then",
                        merge_regions=merge_regions,
                        multi_task=multi_task,
                    ),
                }
            )
        elif not any(
            name in defaults
            for choice in clause["oneOf"]
            for name in choice["required"]
        ):
            conditions.append({"$ref": pointer})
    if conditions:
        result["allOf"] = conditions
    return result


def _when_preset(
    choices: list[tuple[list[str], dict[str, Any]]],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Select the matching preset schema, or the caller's inherited default."""
    return {
        "if": {"required": ["preset"]},
        "then": {
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "preset": {
                                "type": "string",
                                "pattern": _preset_pattern(names),
                            }
                        }
                    },
                    "then": selected,
                }
                for names, selected in choices
            ]
        },
        "else": fallback,
    }


def _model_selector(
    choices: list[tuple[list[str], dict[str, Any]]],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"preset": {"$ref": "#/$defs/model_preset_name"}},
        **_when_preset(choices, fallback),
    }


def with_model_presets(schema: dict[str, Any], multi_task: bool) -> dict[str, Any]:
    """Add raw preset input forms to a generated training schema.

    Parameters
    ----------
    schema : dict
        Training schema generated from the argument definitions, updated in place.
    multi_task : bool
        Whether model entries live in a multi-task ``model_dict``.

    Returns
    -------
    dict
        Schema accepting presets and partial overrides alongside ordinary inputs.
    """
    model = schema["properties"]["model"]
    if multi_task:
        branches = model["properties"]["model_dict"]
        base = branches.pop("items")
    else:
        base = model

    names = sorted(MODEL_PRESETS)
    definitions = {
        "model": base,
        "model_preset_name": {
            "description": "Named model architecture. Explicit model fields override its defaults.",
            "type": "string",
            "anyOf": [{"enum": names}, {"pattern": _preset_pattern(names)}],
        },
    }
    choices = []
    shared_schemas = {}
    for name in names:
        selected = _with_defaults(
            base, MODEL_PRESETS[name], "#/$defs/model", multi_task=multi_task
        )
        key = json.dumps(selected, sort_keys=True)
        if key not in shared_schemas:
            definition = f"model_preset_{len(choices)}"
            definitions[definition] = selected
            shared_schemas[key] = []
            choices.append((shared_schemas[key], {"$ref": f"#/$defs/{definition}"}))
        shared_schemas[key].append(name)
    schema["$defs"] = definitions

    if not multi_task:
        schema["properties"]["model"] = _model_selector(
            choices, {"$ref": "#/$defs/model"}
        )
        return schema

    # Object-valued repeat arguments use additionalProperties, not array items.
    # Non-preset branches retain the existing multi-task schema's permissiveness;
    # their shared references and cascaded fields are resolved at training time.
    branches["additionalProperties"] = _model_selector(choices, {})
    model["properties"]["preset"] = {"$ref": "#/$defs/model_preset_name"}
    inherited = [
        (
            group,
            {
                "allOf": [
                    selected,
                    {
                        "properties": {
                            "model_dict": {
                                "additionalProperties": _model_selector(
                                    choices, selected
                                )
                            }
                        }
                    },
                ]
            },
        )
        for group, selected in choices
    ]
    model.update(_when_preset(inherited, {}))
    return schema
