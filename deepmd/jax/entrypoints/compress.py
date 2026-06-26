# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress a JAX model by tabulating embedding networks."""

import logging
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.common import (
    j_loader,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)
from deepmd.jax.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
)

log = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    return float(np.asarray(value))


def _get_saved_min_nbor_dist(data: dict) -> float | None:
    """Read min_nbor_dist from known serialized-model metadata locations."""
    min_nbor_dist = _to_float(data.get("min_nbor_dist"))
    if min_nbor_dist is not None:
        return min_nbor_dist
    constants = data.get("constants", {})
    return _to_float(constants.get("min_nbor_dist"))


def _get_input_min_nbor_dist(input_file: str, data: dict) -> float | None:
    """Read min_nbor_dist from the serialized data or native HLO constants."""
    min_nbor_dist = _get_saved_min_nbor_dist(data)
    if min_nbor_dist is not None:
        return min_nbor_dist
    if Path(input_file).suffix == ".hlo":
        return _get_saved_min_nbor_dist(load_dp_model(input_file))
    return None


def _compute_min_nbor_dist(training_script: str) -> float:
    jdata = update_deepmd_input(j_loader(training_script))
    type_map = jdata["model"].get("type_map", None)
    train_data = get_data(
        jdata["training"]["training_data"],
        0,
        type_map,
        None,
    )
    return float(UpdateSel().get_min_nbor_dist(train_data))


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
) -> None:
    """Compress a JAX ``.jax``/``.hlo`` model."""
    data = serialize_from_file(input_file)
    model = BaseModel.deserialize(data["model"])

    min_nbor_dist = _to_float(model.get_min_nbor_dist())
    if min_nbor_dist is None:
        min_nbor_dist = _get_input_min_nbor_dist(input_file, data)
    if min_nbor_dist is None:
        log.info(
            "Minimal neighbor distance is not saved in the model, "
            "compute it from the training data."
        )
        if training_script is None:
            raise ValueError(
                "The model does not have a minimum neighbor distance, "
                "so the training script and data must be provided "
                "(via -t,--training-script)."
            )
        min_nbor_dist = _compute_min_nbor_dist(training_script)

    model.min_nbor_dist = float(min_nbor_dist)
    model.enable_compression(
        extrapolate,
        stride,
        stride * 10,
        check_frequency,
    )

    compressed_data = data.copy()
    compressed_data["model"] = model.serialize()
    compressed_data["min_nbor_dist"] = float(min_nbor_dist)
    deserialize_to_file(output, compressed_data)
    log.info("Compressed model saved to %s", output)
