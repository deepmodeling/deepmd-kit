# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fine-tuning rule helpers for the PyTorch backend."""

from typing import (
    Any,
)

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
)
from deepmd.utils.finetune import get_finetune_rule_single as get_finetune_rule_single
from deepmd.utils.finetune import (
    get_finetune_rules_from_model_params,
)

__all__ = ["get_finetune_rule_single", "get_finetune_rules"]


def get_finetune_rules(
    finetune_model: str,
    model_config: dict[str, Any],
    model_branch: str = "",
    change_model_params: bool = True,
) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem]]:
    """Get fine-tuning rules for a single-task or multi-task PyTorch model."""
    state_dict = torch.load(finetune_model, map_location=env.DEVICE, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    return get_finetune_rules_from_model_params(
        state_dict["_extra_state"]["model_params"],
        model_config,
        model_branch=model_branch,
        change_model_params=change_model_params,
        multitask_branch_error=(
            "Multi-task fine-tuning does not support command-line branches chosen!"
            "Please define the 'finetune_head' in each model params!"
        ),
    )
