# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from importlib import (
    import_module,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.multi_task import (
    apply_shared_links,
)
from deepmd.dpmodel.utils.multi_task import (
    preprocess_shared_params as preprocess_shared_params_common,
)
from deepmd.dpmodel.utils.multi_task import (
    set_descriptor_component,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)

log = logging.getLogger(__name__)

# Populate JAX descriptor and fitting registries before class lookup.
import_module("deepmd.jax.descriptor")
import_module("deepmd.jax.fitting.fitting")


def preprocess_shared_params(
    model_config: dict[str, Any],
    require_shared_type_map: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preprocess JAX shared model params and generate sharing links."""
    model_config, shared_links = preprocess_shared_params_common(
        model_config,
        get_class_name,
        require_shared_type_map=require_shared_type_map,
        cascade_defaults=True,
    )
    validate_shared_lora(model_config, shared_links)
    return model_config, shared_links


def validate_shared_lora(
    model_config: dict[str, Any], shared_links: dict[str, Any]
) -> None:
    """Require compatible adapter and freezing policies for shared descriptors."""
    for key, shared in shared_links.items():
        descriptor_links = [
            link
            for link in shared["links"]
            if link["shared_type"].startswith("descriptor")
        ]
        if len(descriptor_links) < 2:
            continue
        base = descriptor_links[0]
        base_lora = model_config["model_dict"][base["model_key"]].get("lora")
        for link in descriptor_links[1:]:
            lora = model_config["model_dict"][link["model_key"]].get("lora")
            if (lora is None) != (base_lora is None):
                raise ValueError(
                    f"Shared descriptor {key!r} requires the same LoRA freezing policy "
                    f"in branches {base['model_key']!r} and {link['model_key']!r}."
                )
            if lora is None or link["shared_level"] != 0:
                continue
            rank = int(lora["rank"])
            base_rank = int(base_lora["rank"])
            alpha = (
                float(lora["alpha"]) if lora.get("alpha") is not None else float(rank)
            )
            base_alpha = (
                float(base_lora["alpha"])
                if base_lora.get("alpha") is not None
                else float(base_rank)
            )
            if (rank, alpha) != (base_rank, base_alpha):
                raise ValueError(
                    f"Shared descriptor {key!r} requires identical LoRA rank and alpha "
                    f"in branches {base['model_key']!r} and {link['model_key']!r}."
                )


def get_class_name(item_key: str, item_params: dict[str, Any]) -> type:
    if item_key == "descriptor":
        return BaseDescriptor.get_class_by_type(item_params.get("type", "se_e2_a"))
    if item_key == "fitting_net":
        return BaseFitting.get_class_by_type(item_params.get("type", "ener"))
    raise RuntimeError(f"Unknown class_name type {item_key}")


def share_jax_model_params(
    models: dict[str, BaseModel],
    shared_links: dict[str, Any],
    *,
    model_key_prob_map: dict[str, float],
    data_stat_protect: float = 1e-2,
    resume: bool = False,
) -> None:
    """Share JAX model parameters following ``preprocess_shared_params`` links."""
    apply_shared_links(
        models,
        shared_links,
        share_descriptor=_share_jax_descriptor,
        share_fitting=_share_jax_fitting,
        model_key_prob_map=model_key_prob_map,
        data_stat_protect=data_stat_protect,
        resume=resume,
        logger=log,
    )


def _share_jax_descriptor(
    link_model: BaseModel,
    link_type: str,
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    resume: bool,
) -> None:
    _share_descriptor_component(
        base_class,
        link_class,
        shared_level,
        model_prob=model_prob,
        resume=resume,
    )
    if shared_level == 0:
        set_descriptor_component(link_model, link_type, base_class)


def _share_jax_fitting(
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    protection: float,
    resume: bool,
) -> None:
    _share_fitting_component(
        base_class,
        link_class,
        shared_level,
        model_prob=model_prob,
        protection=protection,
        resume=resume,
    )


def _share_descriptor_component(
    base_class: Any,
    link_class: Any,
    shared_level: int,
    *,
    model_prob: float,
    resume: bool,
) -> None:
    if type(link_class) is not type(base_class):
        raise AssertionError("Only descriptors of the same type can share params!")
    if shared_level == 0:
        if not resume:
            _merge_descriptor_stats(base_class, link_class, model_prob)
        return
    if shared_level == 1 and hasattr(base_class, "type_embedding"):
        link_class.type_embedding = base_class.type_embedding
        if hasattr(base_class, "charge_spin_embedding"):
            link_class.charge_spin_embedding = base_class.charge_spin_embedding
        return
    raise NotImplementedError(
        f"JAX shared_dict does not support descriptor shared_level={shared_level} "
        f"for {type(base_class).__name__}."
    )


def _merge_descriptor_stats(
    base_class: Any,
    link_class: Any,
    model_prob: float,
) -> None:
    from deepmd.dpmodel.utils.env_mat_stat import (
        merge_env_stat,
    )

    merge_env_stat(base_class, link_class, model_prob)
    for attr in (
        "se_atten",
        "seat",
        "se_ttebd",
        "repinit",
        "repinit_three_body",
        "repformers",
        "repflows",
    ):
        if hasattr(base_class, attr) and hasattr(link_class, attr):
            _merge_descriptor_stats(
                getattr(base_class, attr),
                getattr(link_class, attr),
                model_prob,
            )
    if hasattr(base_class, "descrpt_list") and hasattr(link_class, "descrpt_list"):
        for base_item, link_item in zip(
            base_class.descrpt_list,
            link_class.descrpt_list,
            strict=True,
        ):
            _merge_descriptor_stats(base_item, link_item, model_prob)


def _share_fitting_component(
    base_class: Any,
    link_class: Any,
    shared_level: int,
    *,
    model_prob: float,
    protection: float,
    resume: bool,
) -> None:
    if type(link_class) is not type(base_class):
        raise AssertionError("Only fitting nets of the same type can share params!")
    if shared_level != 0:
        raise NotImplementedError(
            f"JAX shared_dict does not support fitting_net shared_level={shared_level}."
        )
    if not resume:
        _merge_fitting_param_stats(
            base_class,
            link_class,
            model_prob=model_prob,
            protection=protection,
        )
    link_class.nets = base_class.nets
    for attr in (
        "fparam_avg",
        "fparam_inv_std",
        "aparam_avg",
        "aparam_inv_std",
        "default_fparam_tensor",
    ):
        if getattr(base_class, attr, None) is not None:
            setattr(link_class, attr, getattr(base_class, attr))


def _merge_fitting_param_stats(
    base_class: Any,
    link_class: Any,
    *,
    model_prob: float,
    protection: float,
) -> None:
    _merge_one_fitting_stat(
        base_class,
        link_class,
        name="fparam",
        avg_attr="fparam_avg",
        inv_std_attr="fparam_inv_std",
        numb_attr="numb_fparam",
        model_prob=model_prob,
        protection=protection,
    )
    _merge_one_fitting_stat(
        base_class,
        link_class,
        name="aparam",
        avg_attr="aparam_avg",
        inv_std_attr="aparam_inv_std",
        numb_attr="numb_aparam",
        model_prob=model_prob,
        protection=protection,
    )


def _merge_one_fitting_stat(
    base_class: Any,
    link_class: Any,
    *,
    name: str,
    avg_attr: str,
    inv_std_attr: str,
    numb_attr: str,
    model_prob: float,
    protection: float,
) -> None:
    if getattr(base_class, numb_attr, 0) <= 0:
        return
    base_stats = base_class.get_param_stats().get(name, [])
    link_stats = link_class.get_param_stats().get(name, [])
    if not base_stats or not link_stats:
        return
    if len(base_stats) != getattr(base_class, numb_attr):
        raise AssertionError(f"{name} statistics length mismatch!")
    merged = [
        base_stats[ii] + link_stats[ii] * model_prob
        for ii in range(getattr(base_class, numb_attr))
    ]
    avg = np.array([stat.compute_avg() for stat in merged], dtype=np.float64)
    inv_std = 1.0 / np.array(
        [stat.compute_std(protection=protection) for stat in merged],
        dtype=np.float64,
    )
    setattr(base_class, avg_attr, _as_backend_array(getattr(base_class, avg_attr), avg))
    setattr(
        base_class,
        inv_std_attr,
        _as_backend_array(getattr(base_class, inv_std_attr), inv_std),
    )
    base_class._param_stats[name] = merged


def _as_backend_array(reference: Any, value: np.ndarray) -> Any:
    import array_api_compat

    ref_value = getattr(reference, "value", reference)
    xp = array_api_compat.array_namespace(ref_value)
    return xp.asarray(
        value,
        dtype=ref_value.dtype,
        device=array_api_compat.device(ref_value),
    )
