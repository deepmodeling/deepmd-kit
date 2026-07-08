# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress a native DPModel file by tabulating embedding networks."""

import logging

from deepmd.dpmodel.entrypoints.compress_common import (
    enable_model_compression,
    resolve_min_nbor_dist,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)

log = logging.getLogger(__name__)


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
) -> None:
    """Compress a native ``.dp``/``.yaml`` model."""
    model_dict = load_dp_model(input_file)
    model = BaseModel.deserialize(model_dict["model"])

    min_nbor_dist = resolve_min_nbor_dist(
        model,
        [model_dict],
        training_script,
    )
    enable_model_compression(model, min_nbor_dist, stride, extrapolate, check_frequency)

    compressed_model_dict = model_dict.copy()
    compressed_model_dict["model"] = model.serialize()
    compressed_model_dict["min_nbor_dist"] = float(min_nbor_dist)
    save_dp_model(output, compressed_model_dict)
    log.info("Compressed model saved to %s", output)
