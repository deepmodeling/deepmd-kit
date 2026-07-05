# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-task sharing helpers for the TensorFlow 2 backend."""

from __future__ import (
    annotations,
)

import logging
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.utils.env_mat_stat import (
    merge_env_stat,
)
from deepmd.tf2.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.tf2.env import (
    tf,
    xp,
)
from deepmd.tf2.fitting.base_fitting import (
    BaseFitting,
)

log = logging.getLogger(__name__)


def preprocess_shared_params(
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand ``shared_dict`` references and generate runtime sharing links."""
    assert "model_dict" in model_config, "only multi-task model can use this method!"
    supported_types = ["type_map", "descriptor", "fitting_net"]
    shared_dict = model_config.get("shared_dict", {})
    shared_links: dict[str, Any] = {}
    type_map_keys: list[str] = []

    def replace_one_item(
        params_dict: dict[str, Any] | list[Any],
        model_key: str,
        key_type: str,
        key_in_dict: str,
        suffix: str = "",
        index: int | None = None,
    ) -> None:
        shared_type = key_type
        shared_key = key_in_dict
        shared_level = 0
        if ":" in key_in_dict:
            shared_key = key_in_dict.split(":")[0]
            shared_level = int(key_in_dict.split(":")[1])
        assert shared_key in shared_dict, (
            f"Appointed {shared_type} {shared_key} are not in the shared_dict! "
            "Please check the input params."
        )
        if index is None:
            assert isinstance(params_dict, dict)
            params_dict[shared_type] = deepcopy(shared_dict[shared_key])
        else:
            params_dict[index] = deepcopy(shared_dict[shared_key])
        if shared_type == "type_map":
            if key_in_dict not in type_map_keys:
                type_map_keys.append(key_in_dict)
        else:
            if shared_key not in shared_links:
                shared_links[shared_key] = {
                    "type": get_class_name(shared_type, shared_dict[shared_key]),
                    "links": [],
                }
            shared_links[shared_key]["links"].append(
                {
                    "model_key": model_key,
                    "shared_type": shared_type + suffix,
                    "shared_level": shared_level,
                }
            )

    for model_key in model_config["model_dict"]:
        model_params_item = model_config["model_dict"][model_key]
        for item_key in model_params_item:
            if item_key not in supported_types:
                continue
            item_params = model_params_item[item_key]
            if isinstance(item_params, str):
                replace_one_item(model_params_item, model_key, item_key, item_params)
            elif (
                item_key == "descriptor"
                and isinstance(item_params, dict)
                and item_params.get("type", "") == "hybrid"
            ):
                for ii, hybrid_item in enumerate(item_params["list"]):
                    if isinstance(hybrid_item, str):
                        replace_one_item(
                            model_params_item[item_key]["list"],
                            model_key,
                            item_key,
                            hybrid_item,
                            suffix=f"_hybrid_{ii}",
                            index=ii,
                        )

    for shared_key in shared_links:
        shared_links[shared_key]["links"] = sorted(
            shared_links[shared_key]["links"],
            key=lambda x: (
                x["shared_level"]
                - ("spin" in model_config["model_dict"][x["model_key"]]) * 100
            ),
        )
    assert len(type_map_keys) == 1, "Multitask model must have only one type_map!"
    return model_config, shared_links


def get_class_name(item_key: str, item_params: dict[str, Any]) -> type:
    if item_key == "descriptor":
        return BaseDescriptor.get_class_by_type(item_params.get("type", "se_e2_a"))
    if item_key == "fitting_net":
        return BaseFitting.get_class_by_type(item_params.get("type", "ener"))
    raise RuntimeError(f"Unknown class_name type {item_key}")


def sanitize_shared_links(shared_links: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-safe copy of ``shared_links``."""
    if shared_links is None:
        return None
    sanitized: dict[str, Any] = {}
    for shared_key, shared_info in shared_links.items():
        class_type = shared_info.get("type")
        sanitized[shared_key] = {
            "type": getattr(class_type, "__name__", str(class_type)),
            "links": deepcopy(shared_info.get("links", [])),
        }
    return sanitized


def apply_shared_links(
    models: dict[str, Any],
    shared_links: dict[str, Any] | None,
    *,
    model_key_prob_map: dict[str, float] | None = None,
    data_stat_protect: float = 1e-2,
    resume: bool = False,
) -> None:
    """Share TF2 model parameters according to ``shared_links``."""
    if not shared_links:
        return
    if model_key_prob_map is None:
        model_key_prob_map = dict.fromkeys(models, 1.0)

    for shared_item, shared_info in shared_links.items():
        links = shared_info.get("links", [])
        if len(links) < 2:
            continue
        shared_base = links[0]
        class_type_base = shared_base["shared_type"]
        model_key_base = shared_base["model_key"]
        shared_level_base = int(shared_base["shared_level"])
        if "descriptor" in class_type_base:
            base_class = _get_descriptor(models[model_key_base], class_type_base)
            for link_item in links[1:]:
                class_type_link = link_item["shared_type"]
                model_key_link = link_item["model_key"]
                shared_level_link = int(link_item["shared_level"])
                assert shared_level_link >= shared_level_base, (
                    "The shared_links must be sorted by shared_level!"
                )
                assert "descriptor" in class_type_link, (
                    f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                )
                link_class = _get_descriptor(models[model_key_link], class_type_link)
                frac_prob = _model_prob_ratio(
                    model_key_prob_map, model_key_base, model_key_link
                )
                _share_descriptor(
                    models[model_key_link],
                    class_type_link,
                    link_class,
                    base_class,
                    shared_level_link,
                    frac_prob,
                    resume=resume,
                )
                log.warning(
                    "Shared params of %s.%s and %s.%s!",
                    model_key_base,
                    class_type_base,
                    model_key_link,
                    class_type_link,
                )
        else:
            if not hasattr(models[model_key_base].atomic_model, class_type_base):
                continue
            base_class = getattr(models[model_key_base].atomic_model, class_type_base)
            for link_item in links[1:]:
                class_type_link = link_item["shared_type"]
                model_key_link = link_item["model_key"]
                shared_level_link = int(link_item["shared_level"])
                assert shared_level_link >= shared_level_base, (
                    "The shared_links must be sorted by shared_level!"
                )
                assert class_type_base == class_type_link, (
                    f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                )
                link_class = getattr(
                    models[model_key_link].atomic_model, class_type_link
                )
                frac_prob = _model_prob_ratio(
                    model_key_prob_map, model_key_base, model_key_link
                )
                _share_fitting(
                    link_class,
                    base_class,
                    shared_level_link,
                    frac_prob,
                    protection=data_stat_protect,
                    resume=resume,
                )
                log.warning(
                    "Shared params of %s.%s and %s.%s!",
                    model_key_base,
                    class_type_base,
                    model_key_link,
                    class_type_link,
                )


def _model_prob_ratio(
    model_key_prob_map: dict[str, float],
    model_key_base: str,
    model_key_link: str,
) -> float:
    base_prob = float(model_key_prob_map.get(model_key_base, 1.0))
    link_prob = float(model_key_prob_map.get(model_key_link, 1.0))
    if base_prob == 0.0:
        return 1.0
    return link_prob / base_prob


def _get_descriptor(model: Any, shared_type: str) -> Any:
    if shared_type == "descriptor":
        return model.get_descriptor()
    if "hybrid" in shared_type:
        hybrid_index = int(shared_type.split("_")[-1])
        return model.get_descriptor().descrpt_list[hybrid_index]
    raise RuntimeError(f"Unknown class_type {shared_type}!")


def _set_descriptor(model: Any, shared_type: str, descriptor: Any) -> None:
    if shared_type == "descriptor":
        model.atomic_model.descriptor = descriptor
        return
    if "hybrid" in shared_type:
        hybrid_index = int(shared_type.split("_")[-1])
        model.get_descriptor().descrpt_list[hybrid_index] = descriptor
        return
    raise RuntimeError(f"Unknown class_type {shared_type}!")


def _share_descriptor(
    link_model: Any,
    link_type: str,
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    resume: bool,
) -> None:
    assert link_class.__class__ == base_class.__class__, (
        "Only descriptors of the same type can share params!"
    )
    if shared_level == 0:
        if not resume:
            merge_env_stat(base_class, link_class, model_prob)
        _set_descriptor(link_model, link_type, base_class)
        return
    if shared_level == 1 and hasattr(link_class, "type_embedding"):
        link_class.type_embedding = base_class.type_embedding
        return
    raise NotImplementedError(
        f"TF2 descriptor shared level {shared_level} is not supported for "
        f"{link_class.__class__.__name__}."
    )


def _share_fitting(
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    protection: float,
    resume: bool,
) -> None:
    assert link_class.__class__ == base_class.__class__, (
        "Only fitting nets of the same type can share params!"
    )
    if shared_level != 0:
        raise NotImplementedError(
            f"TF2 fitting_net shared level {shared_level} is not supported for "
            f"{link_class.__class__.__name__}."
        )

    _merge_and_share_param_stats(
        link_class,
        base_class,
        "fparam",
        "fparam_avg",
        "fparam_inv_std",
        model_prob,
        protection=protection,
        resume=resume,
    )
    _merge_and_share_param_stats(
        link_class,
        base_class,
        "aparam",
        "aparam_avg",
        "aparam_inv_std",
        model_prob,
        protection=protection,
        resume=resume,
    )
    _share_tf2_state_attrs(
        link_class,
        base_class,
        shared_attr_names={"nets"},
    )


def _merge_and_share_param_stats(
    link_class: Any,
    base_class: Any,
    stat_name: str,
    avg_attr: str,
    inv_std_attr: str,
    model_prob: float,
    *,
    protection: float,
    resume: bool,
) -> None:
    avg_value = getattr(link_class, avg_attr, None)
    inv_std_value = getattr(link_class, inv_std_attr, None)
    if avg_value is None or inv_std_value is None:
        return
    if not resume:
        base_stats = base_class.get_param_stats().get(stat_name, [])
        link_stats = link_class.get_param_stats().get(stat_name, [])
        if base_stats and link_stats:
            assert len(base_stats) == len(link_stats)
            merged = [
                base_stats[ii] + link_stats[ii] * model_prob
                for ii in range(len(base_stats))
            ]
            avg = np.array([stat.compute_avg() for stat in merged], dtype=np.float64)
            inv_std = 1.0 / np.array(
                [stat.compute_std(protection=protection) for stat in merged],
                dtype=np.float64,
            )
            _assign_array_like(base_class, avg_attr, avg)
            _assign_array_like(base_class, inv_std_attr, inv_std)
            base_class._param_stats[stat_name] = merged
    setattr(link_class, avg_attr, getattr(base_class, avg_attr))
    setattr(link_class, inv_std_attr, getattr(base_class, inv_std_attr))


def _assign_array_like(obj: Any, attr: str, value: Any) -> None:
    current = getattr(obj, attr)
    current_xp = array_api_compat.array_namespace(current)
    setattr(
        obj,
        attr,
        current_xp.asarray(
            value,
            dtype=current.dtype,
            device=array_api_compat.device(current),
        ),
    )


def _share_tf2_state_attrs(
    link_class: Any,
    base_class: Any,
    *,
    shared_attr_names: set[str],
) -> None:
    for name in shared_attr_names:
        if not hasattr(link_class, name) or not hasattr(base_class, name):
            continue
        value = getattr(link_class, name)
        if _is_shareable_tf2_state(value):
            setattr(link_class, name, getattr(base_class, name))


def _is_shareable_tf2_state(value: Any) -> bool:
    return isinstance(value, (tf.Module, tf.Variable, tf.Tensor, xp.Array))
