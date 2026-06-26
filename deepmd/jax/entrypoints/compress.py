# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress a JAX model by tabulating embedding networks."""

import logging
from pathlib import (
    Path,
)

from deepmd.dpmodel.entrypoints.compress_common import (
    enable_model_compression,
    resolve_min_nbor_dist,
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

log = logging.getLogger(__name__)


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

    metadata_sources = [data]
    if Path(input_file).suffix == ".hlo":
        metadata_sources.append(load_dp_model(input_file))
    min_nbor_dist = resolve_min_nbor_dist(
        model,
        metadata_sources,
        training_script,
        UpdateSel,
    )
    enable_model_compression(model, min_nbor_dist, stride, extrapolate, check_frequency)

    compressed_data = data.copy()
    compressed_data["model"] = model.serialize()
    compressed_data["min_nbor_dist"] = float(min_nbor_dist)
    deserialize_to_file(output, compressed_data)
    log.info("Compressed model saved to %s", output)
