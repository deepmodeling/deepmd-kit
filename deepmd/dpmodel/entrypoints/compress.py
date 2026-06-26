# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress a native DPModel file by tabulating embedding networks."""

import logging

from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)

log = logging.getLogger(__name__)


def _get_saved_min_nbor_dist(model_dict: dict) -> float | None:
    """Read min_nbor_dist from known native-model metadata locations."""
    min_nbor_dist = model_dict.get("min_nbor_dist")
    if min_nbor_dist is None:
        constants = model_dict.get("constants", {})
        min_nbor_dist = constants.get("min_nbor_dist")
    if min_nbor_dist is None:
        return None
    return float(min_nbor_dist)


def _compute_min_nbor_dist(training_script: str) -> float:
    from deepmd.common import (
        j_loader,
    )
    from deepmd.dpmodel.utils.update_sel import (
        UpdateSel,
    )
    from deepmd.utils.compat import (
        update_deepmd_input,
    )
    from deepmd.utils.data_system import (
        get_data,
    )

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
    """Compress a native ``.dp``/``.yaml`` model."""
    model_dict = load_dp_model(input_file)
    model = BaseModel.deserialize(model_dict["model"])

    min_nbor_dist = model.get_min_nbor_dist()
    if min_nbor_dist is None:
        min_nbor_dist = _get_saved_min_nbor_dist(model_dict)
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

    compressed_model_dict = model_dict.copy()
    compressed_model_dict["model"] = model.serialize()
    compressed_model_dict["min_nbor_dist"] = float(min_nbor_dist)
    save_dp_model(output, compressed_model_dict)
    log.info("Compressed model saved to %s", output)
