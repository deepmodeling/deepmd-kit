# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finetune utilities for the pt_expt backend.

Reuses ``get_finetune_rule_single`` from the pt backend since pt_expt
uses the same checkpoint format (``.pt`` with ``_extra_state.model_params``).
"""

from typing import (
    Any,
)

import torch

from deepmd.pt.utils.finetune import (
    get_finetune_rule_single,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
)


def get_finetune_rules(
    finetune_model: str,
    model_config: dict[str, Any],
    model_branch: str = "",
    change_model_params: bool = True,
) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem]]:
    """Get fine-tuning rules for a single-task pt_expt model.

    Loads a pretrained checkpoint and builds ``FinetuneRuleItem`` objects
    describing how to map types and weights from the pretrained model to
    the new model.

    Parameters
    ----------
    finetune_model : str
        Path to the pretrained ``.pt`` checkpoint.
    model_config : dict
        The model section of the fine-tuning config.
    model_branch : str
        Branch to select from a multi-task pretrained model (command-line).
    change_model_params : bool
        Whether to overwrite descriptor/fitting params from the pretrained model.

    Returns
    -------
    model_config : dict
        Possibly updated model config.
    finetune_links : dict[str, FinetuneRuleItem]
        Fine-tuning rules keyed by ``"Default"``.
    """
    state_dict = torch.load(finetune_model, map_location=DEVICE, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    last_model_params = state_dict["_extra_state"]["model_params"]
    finetune_from_multi_task = "model_dict" in last_model_params

    # pt_expt is single-task only
    if model_branch == "" and "finetune_head" in model_config:
        model_branch = model_config["finetune_head"]
    model_config, finetune_rule = get_finetune_rule_single(
        model_config,
        last_model_params,
        from_multitask=finetune_from_multi_task,
        model_branch="Default",
        model_branch_from=model_branch,
        change_model_params=change_model_params,
    )
    finetune_links: dict[str, FinetuneRuleItem] = {"Default": finetune_rule}
    return model_config, finetune_links
