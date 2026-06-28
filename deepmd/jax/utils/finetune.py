# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fine-tuning config utilities for the JAX backend."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
    get_finetune_rules_from_model_params,
)


def _load_model_params(finetune_model: str) -> dict[str, Any]:
    """Extract model params from a JAX checkpoint."""
    if not finetune_model.endswith(".jax"):
        raise ValueError("JAX fine-tuning currently supports .jax checkpoints only.")
    return serialize_from_file(finetune_model)["model_def_script"]


def get_finetune_rules(
    finetune_model: str,
    model_config: dict[str, Any],
    model_branch: str = "",
    change_model_params: bool = True,
) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem]]:
    """Build JAX fine-tuning rules for single-task or multi-task configs."""
    return get_finetune_rules_from_model_params(
        _load_model_params(finetune_model),
        model_config,
        model_branch=model_branch,
        change_model_params=change_model_params,
    )
