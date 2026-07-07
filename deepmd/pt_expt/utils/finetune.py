# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finetune utilities for the pt_expt backend.

Supports finetuning from both ``.pt`` checkpoints and frozen ``.pte`` models.
"""

from typing import (
    Any,
)

import torch

from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
)
from deepmd.utils.finetune import get_finetune_rule_single as get_finetune_rule_single
from deepmd.utils.finetune import (
    get_finetune_rules_from_model_params,
)

__all__ = ["get_finetune_rule_single", "get_finetune_rules"]


def _is_pte(path: str) -> bool:
    return path.endswith((".pte", ".pt2"))


def _load_model_params(finetune_model: str) -> dict[str, Any]:
    """Extract model_params from a ``.pt`` checkpoint or ``.pte`` frozen model."""
    if _is_pte(finetune_model):
        from deepmd.pt_expt.utils.serialization import (
            serialize_from_file,
        )

        data = serialize_from_file(finetune_model)
        return data["model_def_script"]
    else:
        state_dict = torch.load(finetune_model, map_location=DEVICE, weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        return state_dict["_extra_state"]["model_params"]


def get_finetune_rules(
    finetune_model: str,
    model_config: dict[str, Any],
    model_branch: str = "",
    change_model_params: bool = True,
) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem]]:
    """Get fine-tuning rules for a single-task or multi-task pt_expt model.

    Loads a pretrained ``.pt`` checkpoint or ``.pte`` frozen model and
    builds ``FinetuneRuleItem`` objects describing how to map types and
    weights from the pretrained model to the new model.

    Parameters
    ----------
    finetune_model : str
        Path to the pretrained model (``.pt`` or ``.pte``).
    model_config : dict
        The model section of the fine-tuning config.
    model_branch : str
        Branch to select from a multi-task pretrained model (command-line).
    change_model_params : bool
        Whether to overwrite descriptor/fitting params from the pretrained
        model.  Not supported for ``.pte`` sources.

    Returns
    -------
    model_config : dict
        Possibly updated model config.
    finetune_links : dict[str, FinetuneRuleItem]
        Fine-tuning rules keyed by model branch name (``"Default"`` for
        single-task, or per-branch keys for multi-task).
    """
    last_model_params = _load_model_params(finetune_model)
    return get_finetune_rules_from_model_params(
        last_model_params,
        model_config,
        model_branch=model_branch,
        change_model_params=change_model_params,
        multitask_branch_error=(
            "Multi-task fine-tuning does not support command-line branches chosen! "
            "Please define the 'finetune_head' in each model params!"
        ),
        missing_model_params_error=(
            "Cannot use --use-pretrain-script: the pretrained model does not "
            "contain full model params.  If finetuning from a .pte file, "
            "re-freeze it with the latest code so that model_def_script is embedded."
        ),
    )
