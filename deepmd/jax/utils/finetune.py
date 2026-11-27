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


def merged_initialization_source(model: Any) -> Any:
    """Expose a pretrained function as base weights without mutating its adapters.

    Initialization and fine-tuning construct a fresh target parameterization.
    Folding source adapters first preserves their complete contribution when
    the target has no adapters or uses a different rank. Restart restores the
    original adapter state directly and does not use this transformation.
    """
    from deepmd.jax.descriptor.lora import (
        has_lora,
        merge_lora_into_base,
    )
    from deepmd.jax.env import (
        nnx,
    )

    return merge_lora_into_base(nnx.clone(model)) if has_lora(model) else model


def _load_model_params(finetune_model: str) -> dict[str, Any]:
    """Extract model params from a JAX checkpoint."""
    return serialize_from_file(finetune_model, training=False)["model_def_script"]


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
