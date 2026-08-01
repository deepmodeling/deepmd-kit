# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared helpers for native-model compression entrypoints."""

import logging
from collections.abc import (
    Iterable,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.common import (
    j_loader,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
)

log = logging.getLogger(__name__)


def to_float(value: Any) -> float | None:
    """Convert plain, NumPy, or framework-wrapped scalar values to float."""
    if value is None:
        return None
    value = getattr(value, "value", value)
    return float(np.asarray(value))


def get_saved_min_nbor_dist(data: dict) -> float | None:
    """Read min_nbor_dist from known serialized-model metadata locations."""
    min_nbor_dist = to_float(data.get("min_nbor_dist"))
    if min_nbor_dist is not None:
        return min_nbor_dist
    constants = data.get("constants", {})
    return to_float(constants.get("min_nbor_dist"))


def compute_min_nbor_dist(
    training_script: str,
    update_sel_cls: type[Any] | None = None,
) -> float:
    """Compute min_nbor_dist from the training data."""
    if update_sel_cls is None:
        from deepmd.dpmodel.utils.update_sel import (
            UpdateSel,
        )

        update_sel_cls = UpdateSel

    jdata = update_deepmd_input(j_loader(training_script))
    type_map = jdata["model"].get("type_map", None)
    train_data = get_data(
        jdata["training"]["training_data"],
        0,
        type_map,
        None,
    )
    return float(update_sel_cls().get_min_nbor_dist(train_data))


def resolve_min_nbor_dist(
    model: Any,
    metadata_sources: Iterable[dict],
    training_script: str | None,
    update_sel_cls: type[Any] | None = None,
) -> float:
    """Resolve min_nbor_dist from model, saved metadata, or training data."""
    min_nbor_dist = to_float(model.get_min_nbor_dist())
    if min_nbor_dist is None:
        for metadata in metadata_sources:
            min_nbor_dist = get_saved_min_nbor_dist(metadata)
            if min_nbor_dist is not None:
                break
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
        min_nbor_dist = compute_min_nbor_dist(training_script, update_sel_cls)
    return float(min_nbor_dist)


def enable_model_compression(
    model: Any,
    min_nbor_dist: float,
    stride: float,
    extrapolate: int,
    check_frequency: int,
) -> None:
    """Set compression inputs and enable descriptor tabulation."""
    model.min_nbor_dist = float(min_nbor_dist)
    model.enable_compression(
        table_extrapolate=extrapolate,
        table_stride_1=stride,
        table_stride_2=stride * 10,
        check_frequency=check_frequency,
    )
