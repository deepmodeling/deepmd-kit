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

    if change_model_params and "descriptor" not in last_model_params:
        # For multi-task pretrained, check inside model_dict
        if "model_dict" not in last_model_params or "descriptor" not in next(
            iter(last_model_params["model_dict"].values())
        ):
            raise ValueError(
                "Cannot use --use-pretrain-script: the pretrained model does not "
                "contain full model params.  If finetuning from a .pte file, "
                "re-freeze it with the latest code so that model_def_script is embedded."
            )

    multi_task = "model_dict" in model_config
    finetune_from_multi_task = "model_dict" in last_model_params
    finetune_links: dict[str, FinetuneRuleItem] = {}

    if not multi_task:
        # Single-task target
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
        finetune_links["Default"] = finetune_rule
    else:
        # Multi-task target — mirrors PT's logic
        assert model_branch == "", (
            "Multi-task fine-tuning does not support command-line branches chosen!"
            "Please define the 'finetune_head' in each model params!"
        )
        if not finetune_from_multi_task:
            pretrained_keys = ["Default"]
        else:
            pretrained_keys = list(last_model_params["model_dict"].keys())
        for model_key in model_config["model_dict"]:
            resuming = False
            if (
                "finetune_head" in model_config["model_dict"][model_key]
                and model_config["model_dict"][model_key]["finetune_head"] != "RANDOM"
            ):
                pretrained_key = model_config["model_dict"][model_key]["finetune_head"]
                assert pretrained_key in pretrained_keys, (
                    f"'{pretrained_key}' head chosen to finetune not exist in the pretrained model!"
                    f"Available heads are: {list(pretrained_keys)}"
                )
                model_branch_from = pretrained_key
            elif (
                "finetune_head" not in model_config["model_dict"][model_key]
                and model_key in pretrained_keys
            ):
                # resume — no finetune
                model_branch_from = model_key
                resuming = True
            else:
                # new branch or RANDOM → random fitting
                model_branch_from = "RANDOM"
            model_config["model_dict"][model_key], finetune_rule = (
                get_finetune_rule_single(
                    model_config["model_dict"][model_key],
                    last_model_params,
                    from_multitask=finetune_from_multi_task,
                    model_branch=model_key,
                    model_branch_from=model_branch_from,
                    change_model_params=change_model_params,
                )
            )
            finetune_links[model_key] = finetune_rule
            finetune_links[model_key].resuming = resuming
    return model_config, finetune_links
