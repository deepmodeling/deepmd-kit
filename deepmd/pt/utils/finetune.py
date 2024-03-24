# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from copy import (
    deepcopy,
)

import torch

from deepmd.pt.utils import (
    env,
)

log = logging.getLogger(__name__)


def change_finetune_model_params_single(
    _single_param_target,
    _model_param_pretrained,
    from_multitask=False,
    model_branch="Default",
    model_branch_from="",
):
    single_config = deepcopy(_single_param_target)
    trainable_param = {
        "descriptor": True,
        "fitting_net": True,
    }
    for net_type in trainable_param:
        if net_type in single_config:
            trainable_param[net_type] = single_config[net_type].get("trainable", True)
    if not from_multitask:
        old_type_map, new_type_map = (
            _model_param_pretrained["type_map"],
            single_config["type_map"],
        )
        assert set(new_type_map).issubset(
            old_type_map
        ), "Only support for smaller type map when finetuning or resuming."
        single_config = deepcopy(_model_param_pretrained)
        log.info(
            f"Change the '{model_branch}' model configurations according to the pretrained one..."
        )
        single_config["new_type_map"] = new_type_map
    else:
        model_dict_params = _model_param_pretrained["model_dict"]
        new_fitting = False
        if model_branch_from == "":
            model_branch_chosen = next(iter(model_dict_params.keys()))
            new_fitting = True
            single_config["bias_adjust_mode"] = (
                "set-by-statistic"  # fitting net re-init
            )
            log.warning(
                "The fitting net will be re-init instead of using that in the pretrained model! "
                "The bias_adjust_mode will be set-by-statistic!"
            )
        else:
            model_branch_chosen = model_branch_from
        assert model_branch_chosen in model_dict_params, (
            f"No model branch named '{model_branch_chosen}'! "
            f"Available ones are {list(model_dict_params.keys())}."
        )
        single_config_chosen = deepcopy(model_dict_params[model_branch_chosen])
        old_type_map, new_type_map = (
            single_config_chosen["type_map"],
            single_config["type_map"],
        )
        assert set(new_type_map).issubset(
            old_type_map
        ), "Only support for smaller type map when finetuning or resuming."
        for key_item in ["type_map", "descriptor"]:
            if key_item in single_config_chosen:
                single_config[key_item] = single_config_chosen[key_item]
        if not new_fitting:
            single_config["fitting_net"] = single_config_chosen["fitting_net"]
        log.info(
            f"Change the '{model_branch}' model configurations according to the model branch "
            f"'{model_branch_chosen}' in the pretrained one..."
        )
        single_config["new_type_map"] = new_type_map
        single_config["model_branch_chosen"] = model_branch_chosen
        single_config["new_fitting"] = new_fitting
    for net_type in trainable_param:
        if net_type in single_config:
            single_config[net_type]["trainable"] = trainable_param[net_type]
        else:
            single_config[net_type] = {"trainable": trainable_param[net_type]}
    return single_config


def change_finetune_model_params(finetune_model, model_config, model_branch=""):
    """
    Load model_params according to the pretrained one.
    This function modifies the fine-tuning input in different modes as follows:
    1. Single-task fine-tuning from a single-task pretrained model:
        - Updates the model parameters based on the pretrained model.
    2. Single-task fine-tuning from a multi-task pretrained model:
        - Updates the model parameters based on the selected branch in the pretrained model.
        - The chosen branch can be defined from the command-line or `finetune_head` input parameter.
        - If not defined, model parameters in the fitting network will be randomly initialized.
    3. Multi-task fine-tuning from a single-task pretrained model:
        - Updates model parameters in each branch based on the single branch ('Default') in the pretrained model.
        - If `finetune_head` is not set to 'Default',
          model parameters in the fitting network of the branch will be randomly initialized.
    4. Multi-task fine-tuning from a multi-task pretrained model:
        - Updates model parameters in each branch based on the selected branch in the pretrained model.
        - The chosen branches can be defined from the `finetune_head` input parameter of each model.
        - If `finetune_head` is not defined and the model_key is the same as in the pretrained model,
          it will resume from the model_key branch without fine-tuning.
        - If `finetune_head` is not defined and a new model_key is used,
          model parameters in the fitting network of the branch will be randomly initialized.

    Parameters
    ----------
    finetune_model
        The pretrained model.
    model_config
        The fine-tuning input parameters.
    model_branch
        The model branch chosen in command-line mode, only for single-task fine-tuning.

    Returns
    -------
    model_config:
        Updated model parameters.
    finetune_links:
        Fine-tuning rules in a dict format, with `model_branch`: `model_branch_from` pairs.
        If `model_key` is not in this dict, it will do just resuming instead of fine-tuning.
    """
    multi_task = "model_dict" in model_config
    state_dict = torch.load(finetune_model, map_location=env.DEVICE)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    last_model_params = state_dict["_extra_state"]["model_params"]
    finetune_from_multi_task = "model_dict" in last_model_params
    finetune_links = {}
    if not multi_task:
        # use command-line first
        if model_branch == "" and "finetune_head" in model_config:
            model_branch = model_config["finetune_head"]
        model_config = change_finetune_model_params_single(
            model_config,
            last_model_params,
            from_multitask=finetune_from_multi_task,
            model_branch="Default",
            model_branch_from=model_branch,
        )
        finetune_links["Default"] = (
            model_branch if finetune_from_multi_task else "Default"
        )
    else:
        assert model_branch == "", (
            "Multi-task fine-tuning does not support command-line branches chosen!"
            "Please define the 'finetune_head' in each model params!"
        )
        target_keys = model_config["model_dict"].keys()
        if not finetune_from_multi_task:
            pretrained_keys = ["Default"]
        else:
            pretrained_keys = last_model_params["model_dict"].keys()
        for model_key in target_keys:
            if "finetune_head" in model_config["model_dict"][model_key]:
                pretrained_key = model_config["model_dict"][model_key]["finetune_head"]
                assert pretrained_key in pretrained_keys, (
                    f"'{pretrained_key}' head chosen to finetune not exist in the pretrained model!"
                    f"Available heads are: {list(pretrained_keys)}"
                )
                model_branch_from = pretrained_key
                finetune_links[model_key] = model_branch_from
            elif model_key in pretrained_keys:
                # not do anything if not defined "finetune_head" in heads that exist in the pretrained model
                # this will just do resuming
                model_branch_from = model_key
            else:
                # if not defined "finetune_head" in new heads, the fitting net will bre randomly initialized
                model_branch_from = ""
                finetune_links[model_key] = next(iter(pretrained_keys))
            model_config["model_dict"][model_key] = change_finetune_model_params_single(
                model_config["model_dict"][model_key],
                last_model_params,
                from_multitask=finetune_from_multi_task,
                model_branch=model_key,
                model_branch_from=model_branch_from,
            )
    return model_config, finetune_links
