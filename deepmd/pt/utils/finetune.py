# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from copy import (
    deepcopy,
)

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
)

log = logging.getLogger(__name__)


def get_finetune_rule_single(
    _single_param_target,
    _model_param_pretrained,
    from_multitask=False,
    model_branch="Default",
    model_branch_from="",
    change_model_params=False,
):
    single_config = deepcopy(_single_param_target)
    new_fitting = False
    model_branch_chosen = "Default"

    if not from_multitask:
        single_config_chosen = deepcopy(_model_param_pretrained)
        if model_branch_from == "RANDOM":
            # not ["", "RANDOM"], because single-from-single finetune uses pretrained fitting in default
            new_fitting = True
    else:
        model_dict_params = _model_param_pretrained["model_dict"]
        if model_branch_from in ["", "RANDOM"]:
            model_branch_chosen = next(iter(model_dict_params.keys()))
            new_fitting = True
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
    finetune_rule = FinetuneRuleItem(
        p_type_map=old_type_map,
        type_map=new_type_map,
        model_branch=model_branch_chosen,
        random_fitting=new_fitting,
    )
    if change_model_params:
        trainable_param = {
            "descriptor": single_config.get("descriptor", {}).get("trainable", True),
            "fitting_net": single_config.get("fitting_net", {}).get("trainable", True),
        }
        single_config["descriptor"] = single_config_chosen["descriptor"]
        if not new_fitting:
            single_config["fitting_net"] = single_config_chosen["fitting_net"]
        log.info(
            f"Change the '{model_branch}' model configurations according to the model branch "
            f"'{model_branch_chosen}' in the pretrained one..."
        )
        for net_type in trainable_param:
            if net_type in single_config:
                single_config[net_type]["trainable"] = trainable_param[net_type]
            else:
                single_config[net_type] = {"trainable": trainable_param[net_type]}
    return single_config, finetune_rule


def get_finetune_rules(
    finetune_model, model_config, model_branch="", change_model_params=True
):
    """
    Get fine-tuning rules and (optionally) change the model_params according to the pretrained one.

    This function gets the fine-tuning rules and (optionally) changes input in different modes as follows:
    1. Single-task fine-tuning from a single-task pretrained model:
        - The model will be fine-tuned based on the pretrained model.
        - (Optional) Updates the model parameters based on the pretrained model.
    2. Single-task fine-tuning from a multi-task pretrained model:
        - The model will be fine-tuned based on the selected branch in the pretrained model.
          The chosen branch can be defined from the command-line or `finetune_head` input parameter.
          If not defined, model parameters in the fitting network will be randomly initialized.
        - (Optional) Updates the model parameters based on the selected branch in the pretrained model.
    3. Multi-task fine-tuning from a single-task pretrained model:
        - The model in each branch will be fine-tuned or resumed based on the single branch ('Default') in the pretrained model.
          The chosen branches can be defined from the `finetune_head` input parameter of each branch.
          - If `finetune_head` is defined as 'Default',
            it will be fine-tuned based on the single branch ('Default') in the pretrained model.
          - If `finetune_head` is not defined and the model_key is 'Default',
            it will resume from the single branch ('Default') in the pretrained model without fine-tuning.
          - If `finetune_head` is not defined and the model_key is not 'Default',
            it will be fine-tuned based on the single branch ('Default') in the pretrained model,
            while model parameters in the fitting network of the branch will be randomly initialized.
        - (Optional) Updates model parameters in each branch based on the single branch ('Default') in the pretrained model.
    4. Multi-task fine-tuning from a multi-task pretrained model:
        - The model in each branch will be fine-tuned or resumed based on the chosen branches in the pretrained model.
          The chosen branches can be defined from the `finetune_head` input parameter of each branch.
            - If `finetune_head` is defined as one of the branches in the pretrained model,
              it will be fine-tuned based on the chosen branch in the pretrained model.
            - If `finetune_head` is not defined and the model_key is the same as one of those in the pretrained model,
              it will resume from the model_key branch in the pretrained model without fine-tuning.
            - If `finetune_head` is not defined and a new model_key is used,
              it will be fine-tuned based on the chosen branch in the pretrained model,
              while model parameters in the fitting network of the branch will be randomly initialized.
        - (Optional) Updates model parameters in each branch based on the chosen branches in the pretrained model.

    Parameters
    ----------
    finetune_model
        The pretrained model.
    model_config
        The fine-tuning input parameters.
    model_branch
        The model branch chosen in command-line mode, only for single-task fine-tuning.
    change_model_params
        Whether to change the model parameters according to the pretrained one.

    Returns
    -------
    model_config:
        Updated model parameters.
    finetune_links:
        Fine-tuning rules in a dict format, with `model_branch`: FinetuneRuleItem pairs.
    """
    multi_task = "model_dict" in model_config
    state_dict = torch.load(finetune_model, map_location=env.DEVICE, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    last_model_params = state_dict["_extra_state"]["model_params"]
    finetune_from_multi_task = "model_dict" in last_model_params
    finetune_links = {}
    if not multi_task:
        # use command-line first
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
                # not do anything if not defined "finetune_head" in heads that exist in the pretrained model
                # this will just do resuming
                model_branch_from = model_key
                resuming = True
            else:
                # if not defined "finetune_head" in new heads or "finetune_head" is "RANDOM", the fitting net will bre randomly initialized
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
