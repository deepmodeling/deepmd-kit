# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import deepmd.jax.descriptor as _jax_descriptor  # noqa: F401
import deepmd.jax.fitting.fitting as _jax_fitting  # noqa: F401
from deepmd.dpmodel.utils.multi_task import (
    preprocess_shared_params as preprocess_shared_params_common,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)


def preprocess_shared_params(
    model_config: dict[str, Any],
    require_shared_type_map: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preprocess JAX shared model params and generate sharing links."""
    return preprocess_shared_params_common(
        model_config,
        get_class_name,
        require_shared_type_map=require_shared_type_map,
        cascade_defaults=True,
    )


def get_class_name(item_key: str, item_params: dict[str, Any]) -> type:
    if item_key == "descriptor":
        return BaseDescriptor.get_class_by_type(item_params.get("type", "se_e2_a"))
    if item_key == "fitting_net":
        return BaseFitting.get_class_by_type(item_params.get("type", "ener"))
    raise RuntimeError(f"Unknown class_name type {item_key}")
