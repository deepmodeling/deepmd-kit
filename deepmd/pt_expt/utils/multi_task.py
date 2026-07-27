# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.utils.multi_task import (
    preprocess_shared_params as preprocess_shared_params_common,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.fitting import (
    BaseFitting,
)


def preprocess_shared_params(
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return preprocess_shared_params_common(model_config, get_class_name)


def get_class_name(item_key: str, item_params: dict[str, Any]) -> type:
    if item_key == "descriptor":
        return BaseDescriptor.get_class_by_type(item_params.get("type", "se_e2_a"))
    elif item_key == "fitting_net":
        return BaseFitting.get_class_by_type(item_params.get("type", "ener"))
    else:
        raise RuntimeError(f"Unknown class_name type {item_key}")
