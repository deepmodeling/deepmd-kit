# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.utils.multi_task import (
    preprocess_shared_params as preprocess_shared_params_common,
)
from deepmd.pt.model.descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task import (
    BaseFitting,
)


def preprocess_shared_params(
    model_config: dict[str, Any],
    require_shared_type_map: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preprocess shared model params and generate links for parameter sharing.

    Args:
        model_config: Model params containing ``model_dict`` and optional
            ``shared_dict``.
        require_shared_type_map: Whether every branch must define the same
            ordered ``type_map`` after resolving shared references. Linear
            models set this to false and validate their sub-model type maps
            independently.

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
    For example, if one has `model_config` like this:
    "model": {
        "shared_dict": {
            "my_type_map": ["foo", "bar"],
            "my_des1": {
                "type": "se_e2_a",
                "neuron": [10, 20, 40]
                },
        },
        "model_dict": {
            "model_1": {
                "type_map": "my_type_map",
                "descriptor": "my_des1",
                "fitting_net": {
                    "neuron": [100, 100, 100]
                }
            },
            "model_2": {
                "type_map": "my_type_map",
                "descriptor": "my_des1",
                "fitting_net": {
                    "neuron": [100, 100, 100]
                }
            }
            "model_3": {
                "type_map": "my_type_map",
                "descriptor": "my_des1:1",
                "fitting_net": {
                    "neuron": [100, 100, 100]
                }
            }
        }
    }
    The above config will init three model branches named `model_1` and `model_2` and `model_3`,
    in which:
        - `model_2` and `model_3` will have the same `type_map` as that in `model_1`.
        - `model_2` will share all the parameters of `descriptor` with `model_1`,
        while `model_3` will share part of parameters of `descriptor` with `model_1`
        on human-defined share-level `1` (default is `0`, meaning share all the parameters).
        - `model_1`, `model_2` and `model_3` have three different `fitting_net`s.
    The returned `model_config` will automatically fulfill the input `model_config` as if there's no sharing,
    and the `shared_links` will keep all the sharing information with looking:
    {
    'my_des1': {
        'type': 'DescrptSeA',
        'links': [
            {'model_key': 'model_1',
            'shared_type': 'descriptor',
            'shared_level': 0},
            {'model_key': 'model_2',
            'shared_type': 'descriptor',
            'shared_level': 0},
            {'model_key': 'model_3',
            'shared_type': 'descriptor',
            'shared_level': 1}
            ]
        }
    }
    Any key placed directly under ``model`` other than ``model_dict`` /
    ``shared_dict`` is lowered into every branch before shared references are
    resolved (explicit branch values win), so model-wide switches can be
    written once at the top level.
    """
    return preprocess_shared_params_common(
        model_config,
        get_class_name,
        require_shared_type_map=require_shared_type_map,
        cascade_defaults=True,
    )


def get_class_name(item_key: str, item_params: dict[str, Any]) -> type:
    if item_key == "descriptor":
        return BaseDescriptor.get_class_by_type(item_params.get("type", "se_e2_a"))
    elif item_key == "fitting_net":
        return BaseFitting.get_class_by_type(item_params.get("type", "ener"))
    else:
        raise RuntimeError(f"Unknown class_name type {item_key}")
