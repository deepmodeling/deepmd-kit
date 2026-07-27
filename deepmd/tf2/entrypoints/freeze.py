# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze utilities for the TensorFlow 2 backend."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
)
from deepmd.tf2.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)
from deepmd.utils.model_branch_dict import (
    get_model_dict,
)


def freeze(
    *,
    checkpoint_folder: str,
    output: str,
    head: str | None = None,
    **kwargs: Any,
) -> None:
    """Freeze a TF2 training checkpoint into a TensorFlow SavedModel."""
    del kwargs
    output = format_model_suffix(
        output,
        preferred_backend="tf2",
        strict_prefer=True,
    )
    data = serialize_from_file(checkpoint_folder)
    data = select_model_branch(data, head=head)
    deserialize_to_file(output, data)


def select_model_branch(
    data: dict[str, Any], head: str | None = None
) -> dict[str, Any]:
    """Select one branch from a single-task or multi-task serialized payload."""
    model_def_script = data["model_def_script"]
    if "model_dict" not in model_def_script:
        if head not in (None, "", DEFAULT_TASK_KEY):
            raise ValueError(
                f"Single-task TF2 checkpoints do not have a head named {head!r}."
            )
        return data

    if head in (None, ""):
        raise ValueError(
            "Multi-task TF2 checkpoints require --head/--model-branch to select "
            "which model branch to freeze."
        )
    model_alias_dict, _ = get_model_dict(model_def_script["model_dict"])
    if head not in model_alias_dict:
        raise ValueError(
            f"No model branch or alias named {head!r}. Available branches are "
            f"{list(model_def_script['model_dict'])}."
        )
    resolved_head = model_alias_dict[head]
    selected = data.copy()
    selected["model"] = data["model"]["model_dict"][resolved_head]
    selected["model_def_script"] = model_def_script["model_dict"][resolved_head]
    min_nbor_dist = data.get("min_nbor_dist")
    if isinstance(min_nbor_dist, dict):
        selected["min_nbor_dist"] = min_nbor_dist.get(resolved_head)
    return selected
