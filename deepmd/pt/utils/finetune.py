# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import torch

from deepmd.pt.utils import (
    env,
)

log = logging.getLogger(__name__)


def change_finetune_model_params(
    ckpt, finetune_model, model_config, multi_task=False, model_branch=""
):
    """Load model_params according to the pretrained one.

    Args:
    - ckpt & finetune_model: origin model.
    - config: Read from json file.
    """
    # TODO need support for multitask mode
    if finetune_model is not None:
        state_dict = torch.load(finetune_model, map_location=env.DEVICE)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        last_model_params = state_dict["_extra_state"]["model_params"]
        finetune_multi_task = "model_dict" in last_model_params
        trainable_param = {
            "descriptor": True,
            "fitting_net": True,
        }
        for net_type in trainable_param:
            if net_type in model_config:
                trainable_param[net_type] = model_config[net_type].get(
                    "trainable", True
                )
        if not finetune_multi_task:
            old_type_map, new_type_map = (
                last_model_params["type_map"],
                model_config["type_map"],
            )
            assert set(new_type_map).issubset(
                old_type_map
            ), "Only support for smaller type map when finetuning or resuming."
            model_config = last_model_params
            log.info(
                "Change the model configurations according to the pretrained one..."
            )
            model_config["new_type_map"] = new_type_map
        else:
            model_config["finetune_multi_task"] = finetune_multi_task
            model_dict_params = last_model_params["model_dict"]
            new_fitting = False
            if model_branch == "":
                model_branch_chosen = next(iter(model_dict_params.keys()))
                new_fitting = True
                model_config["bias_shift"] = "statistic"  # fitting net re-init
                log.warning(
                    "The fitting net will be re-init instead of using that in the pretrained model! "
                    "The bias_shift will be statistic!"
                )
            else:
                model_branch_chosen = model_branch
            assert model_branch_chosen in model_dict_params, (
                f"No model branch named '{model_branch_chosen}'! "
                f"Available ones are {list(model_dict_params.keys())}."
            )
            old_type_map, new_type_map = (
                model_dict_params[model_branch_chosen]["type_map"],
                model_config["type_map"],
            )
            assert set(new_type_map).issubset(
                old_type_map
            ), "Only support for smaller type map when finetuning or resuming."
            for key_item in ["type_map", "descriptor"]:
                if key_item in model_dict_params[model_branch_chosen]:
                    model_config[key_item] = model_dict_params[model_branch_chosen][
                        key_item
                    ]
            if not new_fitting:
                model_config["fitting_net"] = model_dict_params[model_branch_chosen][
                    "fitting_net"
                ]
            log.info(
                f"Change the model configurations according to the model branch "
                f"{model_branch_chosen} in the pretrained one..."
            )
            model_config["new_type_map"] = new_type_map
            model_config["model_branch_chosen"] = model_branch_chosen
            model_config["new_fitting"] = new_fitting
        for net_type in trainable_param:
            if net_type in model_config:
                model_config[net_type]["trainable"] = trainable_param[net_type]
            else:
                model_config[net_type] = {"trainable": trainable_param[net_type]}
    return model_config
