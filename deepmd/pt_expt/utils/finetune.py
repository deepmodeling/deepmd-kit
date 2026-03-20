# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finetune utilities for the pt_expt backend.

Supports finetuning from both ``.pt`` checkpoints and frozen ``.pte`` models.
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


def _is_pte(path: str) -> bool:
    return path.endswith((".pte", ".pt2"))


def _load_model_params(finetune_model: str) -> dict[str, Any]:
    """Extract model_params from a ``.pt`` checkpoint or ``.pte`` frozen model."""
    if _is_pte(finetune_model):
        from deepmd.pt_expt.utils.serialization import (
            serialize_from_file,
        )

        data = serialize_from_file(finetune_model)
        # Prefer embedded model_params (full config); fall back to
        # a minimal dict with just type_map for older .pte files.
        if "model_params" in data:
            return data["model_params"]
        return {"type_map": data["model"]["type_map"]}
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
    """Get fine-tuning rules for a single-task pt_expt model.

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
        Fine-tuning rules keyed by ``"Default"``.
    """
    last_model_params = _load_model_params(finetune_model)

    if change_model_params and "descriptor" not in last_model_params:
        raise ValueError(
            "Cannot use --use-pretrain-script: the pretrained model does not "
            "contain full model params.  If finetuning from a .pte file, "
            "re-freeze it with the latest code so that model_params is embedded."
        )

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
