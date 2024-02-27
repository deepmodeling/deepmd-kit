# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)

from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptSeA,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.model.task import (
    EnergyFittingNet,
    EnergyFittingNetDirect,
    FittingNetAttenLcc,
)


def preprocess_shared_params(model_config):
    """Preprocess the model params for multitask model, and generate the links dict for further sharing.

    Args:
        model_config: Model params of multitask model.

    Returns
    -------
    model_config: Preprocessed model params of multitask model.
        Those string names are replaced with real params in `shared_dict` of model params.
    shared_links: Dict of link infos for further sharing.
        Each item, whose key must be in `shared_dict`, is a dict with following keys:
        - "type": The real class type of this item.
        - "links": List of shared settings, each sub-item is a dict with following keys:
            - "model_key": Model key in the `model_dict` to share this item.
            - "shared_type": Type of this shard item.
            - "shared_level": Shared level (int) of this item in this model.
                Lower for more params to share, 0 means to share all params in this item.
            This list are sorted by "shared_level".
    """
    assert "model_dict" in model_config, "only multi-task model can use this method!"
    supported_types = ["type_map", "type_embedding", "descriptor", "fitting_net"]
    shared_dict = model_config.get("shared_dict", {})
    shared_links = {}
    type_map_keys = []

    def replace_one_item(params_dict, key_type, key_in_dict, suffix="", index=None):
        shared_type = key_type
        shared_key = key_in_dict
        shared_level = 0
        if ":" in key_in_dict:
            shared_key = key_in_dict.split(":")[0]
            shared_level = int(key_in_dict.split(":")[1])
        assert (
            shared_key in shared_dict
        ), f"Appointed {shared_type} {shared_key} are not in the shared_dict! Please check the input params."
        if index is None:
            params_dict[shared_type] = deepcopy(shared_dict[shared_key])
        else:
            params_dict[index] = deepcopy(shared_dict[shared_key])
        if shared_type == "type_map":
            if key_in_dict not in type_map_keys:
                type_map_keys.append(key_in_dict)
        else:
            if shared_key not in shared_links:
                class_name = get_class_name(shared_type, shared_dict[key_in_dict])
                shared_links[shared_key] = {"type": class_name, "links": []}
            link_item = {
                "model_key": model_key,
                "shared_type": shared_type + suffix,
                "shared_level": shared_level,
            }
            shared_links[shared_key]["links"].append(link_item)

    for model_key in model_config["model_dict"]:
        model_params_item = model_config["model_dict"][model_key]
        for item_key in model_params_item:
            if item_key in supported_types:
                item_params = model_params_item[item_key]
                if isinstance(item_params, str):
                    replace_one_item(model_params_item, item_key, item_params)
                elif item_params.get("type", "") == "hybrid":
                    for ii, hybrid_item in enumerate(item_params["list"]):
                        if isinstance(hybrid_item, str):
                            replace_one_item(
                                model_params_item[item_key]["list"],
                                item_key,
                                hybrid_item,
                                suffix=f"_hybrid_{ii}",
                                index=ii,
                            )
    for shared_key in shared_links:
        shared_links[shared_key]["links"] = sorted(
            shared_links[shared_key]["links"], key=lambda x: x["shared_level"]
        )
    assert len(type_map_keys) == 1, "Multitask model must have only one type_map!"
    return model_config, shared_links


def get_class_name(item_key, item_params):
    if item_key == "type_embedding":
        return TypeEmbedNet.__name__
    elif item_key == "descriptor":
        item_type = item_params.get("type", "se_e2_a")
        if item_type == "se_e2_a":
            return DescrptSeA.__name__
        elif item_type in ["se_atten", "dpa1"]:
            return DescrptDPA1.__name__
        elif item_type in ["dpa2"]:
            return DescrptDPA2.__name__
        # todo add support for other combination
        # elif item_type == "gaussian_lcc":
        #     return DescrptGaussianLcc.__name__
        # elif item_type == "hybrid":
        #     return DescrptHybrid.__name__
        else:
            raise RuntimeError(f"Unknown descriptor type {item_type}")
    elif item_key == "fitting_net":
        item_type = item_params.get("type", "ener")
        if item_type == "ener":
            return EnergyFittingNet.__name__
        elif item_type in ["direct_force", "direct_force_ener"]:
            return EnergyFittingNetDirect.__name__
        elif item_type == "atten_vec_lcc":
            return FittingNetAttenLcc.__name__
        else:
            raise RuntimeError(f"Unknown fitting_net type {item_type}")
    else:
        raise RuntimeError(f"Unknown class_name type {item_key}")
