# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import logging
from typing import (
    Any,
    Dict,
)

from deepmd.tf.utils.errors import (
    GraphWithoutTensorError,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name,
)

log = logging.getLogger(__name__)


def replace_model_params_with_frz_multi_model(
    jdata: Dict[str, Any], pretrained_model: str
):
    """Replace the model params in input script according to pretrained frozen multi-task united model.

    Parameters
    ----------
    jdata : Dict[str, Any]
        input script
    pretrained_model : str
        filename of the pretrained frozen multi-task united model
    """
    # Get the input script from the pretrained model
    try:
        t_jdata = get_tensor_by_name(pretrained_model, "train_attr/training_script")
    except GraphWithoutTensorError as e:
        raise RuntimeError(
            "The input frozen pretrained model: %s has no training script, "
            "which is not supported to perform multi-task training. "
            "Please use the model pretrained with v2.1.5 or higher version of DeePMD-kit."
            % input
        ) from e
    pretrained_jdata = json.loads(t_jdata)

    # Check the model type
    assert "fitting_net_dict" in pretrained_jdata["model"], (
        "The multi-task init process only supports models trained in multi-task mode and frozen into united model!"
        "Please use '--united-model' argument in 'dp freeze' command."
    )

    # Check the type map
    pretrained_type_map = pretrained_jdata["model"]["type_map"]
    cur_type_map = jdata["model"].get("type_map", [])
    out_line_type = []
    for i in cur_type_map:
        if i not in pretrained_type_map:
            out_line_type.append(i)
    assert not out_line_type, (
        f"{out_line_type!s} type(s) not contained in the pretrained model! "
        "Please choose another suitable one."
    )
    if cur_type_map != pretrained_type_map:
        log.info(
            "Change the type_map from {} to {}.".format(
                str(cur_type_map), str(pretrained_type_map)
            )
        )
        jdata["model"]["type_map"] = pretrained_type_map

    # Change model configurations
    pretrained_fitting_keys = sorted(
        pretrained_jdata["model"]["fitting_net_dict"].keys()
    )
    cur_fitting_keys = sorted(jdata["model"]["fitting_net_dict"].keys())
    newly_added_fittings = set(cur_fitting_keys) - set(pretrained_fitting_keys)
    reused_fittings = set(cur_fitting_keys) - newly_added_fittings
    log.info("Change the model configurations according to the pretrained one...")

    for config_key in ["type_embedding", "descriptor", "fitting_net_dict"]:
        if (
            config_key not in jdata["model"].keys()
            and config_key in pretrained_jdata["model"].keys()
        ):
            log.info(
                "Add the '{}' from pretrained model: {}.".format(
                    config_key, str(pretrained_jdata["model"][config_key])
                )
            )
            jdata["model"][config_key] = pretrained_jdata["model"][config_key]
        elif (
            config_key == "type_embedding"
            and config_key in jdata["model"].keys()
            and config_key not in pretrained_jdata["model"].keys()
        ):
            # 'type_embedding' can be omitted using 'se_atten' descriptor, and the activation_function will be None.
            cur_para = jdata["model"].pop(config_key)
            if "trainable" in cur_para and not cur_para["trainable"]:
                jdata["model"][config_key] = {
                    "trainable": False,
                    "activation_function": "None",
                }
                log.info("The type_embeddings from pretrained model will be frozen.")
        elif config_key == "fitting_net_dict":
            if reused_fittings:
                log.info(
                    f"These fitting nets will use the configurations from pretrained frozen model : {reused_fittings}."
                )
                for fitting_key in reused_fittings:
                    _change_sub_config(
                        jdata["model"][config_key],
                        pretrained_jdata["model"][config_key],
                        fitting_key,
                    )
            if newly_added_fittings:
                log.info(
                    f"These fitting nets will be initialized from scratch : {newly_added_fittings}."
                )
        elif (
            config_key in jdata["model"].keys()
            and config_key in pretrained_jdata["model"].keys()
            and jdata["model"][config_key] != pretrained_jdata["model"][config_key]
        ):
            _change_sub_config(jdata["model"], pretrained_jdata["model"], config_key)

    # Change other multi-task configurations
    log.info("Change the training configurations according to the pretrained one...")
    for config_key in ["loss_dict", "training/data_dict"]:
        cur_jdata = jdata
        target_jdata = pretrained_jdata
        for sub_key in config_key.split("/"):
            cur_jdata = cur_jdata[sub_key]
            target_jdata = target_jdata[sub_key]
        for fitting_key in reused_fittings:
            if fitting_key not in cur_jdata:
                target_para = target_jdata[fitting_key]
                cur_jdata[fitting_key] = target_para
                log.info(
                    f"Add '{config_key}/{fitting_key}' configurations from the pretrained frozen model."
                )

    # learning rate dict keep backward compatibility
    config_key = "learning_rate_dict"
    single_config_key = "learning_rate"
    cur_jdata = jdata
    target_jdata = pretrained_jdata
    if (single_config_key not in cur_jdata) and (config_key in cur_jdata):
        cur_jdata = cur_jdata[config_key]
        if config_key in target_jdata:
            target_jdata = target_jdata[config_key]
            for fitting_key in reused_fittings:
                if fitting_key not in cur_jdata:
                    target_para = target_jdata[fitting_key]
                    cur_jdata[fitting_key] = target_para
                    log.info(
                        f"Add '{config_key}/{fitting_key}' configurations from the pretrained frozen model."
                    )
        else:
            for fitting_key in reused_fittings:
                if fitting_key not in cur_jdata:
                    cur_jdata[fitting_key] = {}
                    log.info(
                        f"Add '{config_key}/{fitting_key}' configurations as default."
                    )

    return jdata


def _change_sub_config(jdata: Dict[str, Any], src_jdata: Dict[str, Any], sub_key: str):
    target_para = src_jdata[sub_key]
    cur_para = jdata[sub_key]
    # keep some params that are irrelevant to model structures (need to discuss) TODO
    if "trainable" in cur_para.keys():
        target_para["trainable"] = cur_para["trainable"]
    log.info(f"Change the '{sub_key}' from {cur_para!s} to {target_para!s}.")
    jdata[sub_key] = target_para
