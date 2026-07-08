# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress TensorFlow 2 checkpoints by tabulating embedding networks."""

from __future__ import (
    annotations,
)

import logging
from typing import (
    Any,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.dpmodel.entrypoints.compress_common import (
    enable_model_compression,
    resolve_min_nbor_dist,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.tf2.entrypoints.freeze import (
    select_model_branch,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)

log = logging.getLogger(__name__)


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
    head: str | None = None,
    **kwargs: Any,
) -> None:
    """Compress a TF2 training checkpoint and export a SavedModel."""
    del kwargs
    output = format_model_suffix(
        output,
        preferred_backend="tf2",
        strict_prefer=True,
    )
    data = serialize_from_file(input_file)
    data = select_model_branch(data, head=head)
    model = BaseModel.deserialize(data["model"])
    min_nbor_dist = resolve_min_nbor_dist(
        model,
        [data],
        training_script,
        UpdateSel,
    )
    enable_model_compression(
        model,
        min_nbor_dist,
        stride,
        extrapolate,
        check_frequency,
    )

    compressed_data = data.copy()
    compressed_data["model"] = model.serialize()
    compressed_data["min_nbor_dist"] = float(min_nbor_dist)
    deserialize_to_file(output, compressed_data)
    log.info("Compressed TF2 model saved to %s", output)
