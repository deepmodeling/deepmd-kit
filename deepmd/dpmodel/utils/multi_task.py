# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-neutral helpers for multi-task shared-parameter wiring."""

import logging
from collections.abc import (
    Callable,
)
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

log = logging.getLogger(__name__)


def cascade_top_level_defaults(model_config: dict[str, Any]) -> None:
    """Lower model-wide entries into each multi-task branch in-place."""
    reserved_top_level = ("model_dict", "shared_dict")
    top_level_defaults = {
        k: deepcopy(v) for k, v in model_config.items() if k not in reserved_top_level
    }
    for branch in model_config["model_dict"].values():
        for k, v in top_level_defaults.items():
            branch.setdefault(k, deepcopy(v))
    for k in top_level_defaults:
        model_config.pop(k, None)


def preprocess_shared_params(
    model_config: dict[str, Any],
    get_class_name: Callable[[str, dict[str, Any]], type],
    *,
    require_shared_type_map: bool = True,
    cascade_defaults: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand ``shared_dict`` references and generate runtime sharing links.

    Parameters
    ----------
    model_config : dict[str, Any]
        Multi-task model configuration containing ``model_dict`` and an optional
        ``shared_dict``. String references to shared descriptors, fitting
        networks, and type maps are replaced in-place by their configurations.
    get_class_name : Callable[[str, dict[str, Any]], type]
        Backend-specific callback that resolves the class for a shared
        descriptor or fitting-network configuration.
    require_shared_type_map : bool, default=True
        Whether exactly one shared type-map reference is required. If false, at
        most one shared type-map reference is allowed.
    cascade_defaults : bool, default=False
        Whether to copy non-reserved top-level model options into each model
        branch that does not define them, then remove those options from the top
        level.

    Returns
    -------
    model_config : dict[str, Any]
        The input configuration with shared references expanded.
    shared_links : dict[str, Any]
        Sharing metadata keyed by entries in ``shared_dict``. Each entry records
        the resolved class and the model components linked to it, sorted by
        sharing level.
    """
    assert "model_dict" in model_config, "only multi-task model can use this method!"
    if cascade_defaults:
        cascade_top_level_defaults(model_config)

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
                isinstance(item_params, dict)
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
    if require_shared_type_map:
        assert len(type_map_keys) == 1, "Multitask model must have only one type_map!"
    else:
        assert len(type_map_keys) <= 1, "Shared params must have at most one type_map!"
    return model_config, shared_links


def apply_shared_links(
    models: dict[str, Any],
    shared_links: dict[str, Any] | None,
    *,
    share_descriptor: Callable[..., None],
    share_fitting: Callable[..., None],
    model_key_prob_map: dict[str, float] | None = None,
    data_stat_protect: float = 1e-2,
    resume: bool = False,
    get_descriptor: Callable[[Any, str], Any] | None = None,
    get_fitting: Callable[[Any, str], Any | None] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Apply sharing links to constructed models with backend-specific hooks.

    Parameters
    ----------
    models : dict[str, Any]
        Constructed models keyed by the model names used in ``shared_links``.
    shared_links : dict[str, Any] or None
        Sharing metadata generated by :func:`preprocess_shared_params`. No work
        is performed when it is empty or ``None``.
    share_descriptor : Callable[..., None]
        Backend-specific callback that shares a linked descriptor with its base
        descriptor.
    share_fitting : Callable[..., None]
        Backend-specific callback that shares a linked fitting network with its
        base fitting network.
    model_key_prob_map : dict[str, float] or None, default=None
        Sampling probabilities keyed by model name. The linked-to-base
        probability ratio is passed to the sharing callbacks to weight merged
        statistics. Equal probabilities are used when omitted.
    data_stat_protect : float, default=1e-2
        Protection value passed to the fitting-network sharing callback when
        statistics are merged.
    resume : bool, default=False
        Whether sharing is being restored from a checkpoint. Backend callbacks
        use this flag to skip merging statistics again.
    get_descriptor : Callable[[Any, str], Any] or None, default=None
        Optional accessor for a descriptor or hybrid sub-descriptor. The common
        model accessor is used when omitted.
    get_fitting : Callable[[Any, str], Any or None] or None, default=None
        Optional accessor for a fitting component. The common atomic-model
        accessor is used when omitted.
    logger : logging.Logger or None, default=None
        Logger used to report parameter-sharing operations.

    Returns
    -------
    None
        The components in ``models`` are shared in-place.
    """
    if not shared_links:
        return
    if model_key_prob_map is None:
        model_key_prob_map = dict.fromkeys(models, 1.0)
    if get_descriptor is None:
        get_descriptor = get_descriptor_component
    if get_fitting is None:
        get_fitting = get_atomic_model_attr
    if logger is None:
        logger = log

    for shared_info in shared_links.values():
        links = shared_info.get("links", [])
        if len(links) < 2:
            continue
        shared_base = links[0]
        class_type_base = shared_base["shared_type"]
        model_key_base = shared_base["model_key"]
        shared_level_base = int(shared_base["shared_level"])

        if "descriptor" in class_type_base:
            base_class = get_descriptor(models[model_key_base], class_type_base)
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
                link_model = models[model_key_link]
                link_class = get_descriptor(link_model, class_type_link)
                share_descriptor(
                    link_model,
                    class_type_link,
                    link_class,
                    base_class,
                    shared_level_link,
                    _model_prob_ratio(
                        model_key_prob_map, model_key_base, model_key_link
                    ),
                    resume=resume,
                )
                logger.warning(
                    "Shared params of %s.%s and %s.%s!",
                    model_key_base,
                    class_type_base,
                    model_key_link,
                    class_type_link,
                )
            continue

        base_class = get_fitting(models[model_key_base], class_type_base)
        if base_class is None:
            continue
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
            link_class = get_fitting(models[model_key_link], class_type_link)
            if link_class is None:
                continue
            share_fitting(
                link_class,
                base_class,
                shared_level_link,
                _model_prob_ratio(model_key_prob_map, model_key_base, model_key_link),
                protection=data_stat_protect,
                resume=resume,
            )
            logger.warning(
                "Shared params of %s.%s and %s.%s!",
                model_key_base,
                class_type_base,
                model_key_link,
                class_type_link,
            )


def get_descriptor_component(model: Any, shared_type: str) -> Any:
    """Get a model descriptor or a hybrid sub-descriptor by shared type."""
    if shared_type == "descriptor":
        return model.get_descriptor()
    if "hybrid" in shared_type:
        hybrid_index = int(shared_type.split("_")[-1])
        return model.get_descriptor().descrpt_list[hybrid_index]
    raise RuntimeError(f"Unknown class_type {shared_type}!")


def set_descriptor_component(model: Any, shared_type: str, descriptor: Any) -> None:
    """Set a model descriptor or a hybrid sub-descriptor by shared type."""
    if shared_type == "descriptor":
        model.atomic_model.descriptor = descriptor
        return
    if "hybrid" in shared_type:
        hybrid_index = int(shared_type.split("_")[-1])
        model.get_descriptor().descrpt_list[hybrid_index] = descriptor
        return
    raise RuntimeError(f"Unknown class_type {shared_type}!")


def get_atomic_model_attr(model: Any, attr: str) -> Any | None:
    """Get an attribute from ``model.atomic_model`` if it exists."""
    if hasattr(model.atomic_model, attr):
        return getattr(model.atomic_model, attr)
    return None


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


def _model_prob_ratio(
    model_key_prob_map: dict[str, float],
    model_key_base: str,
    model_key_link: str,
) -> float:
    return float(model_key_prob_map[model_key_link]) / float(
        model_key_prob_map[model_key_base]
    )
