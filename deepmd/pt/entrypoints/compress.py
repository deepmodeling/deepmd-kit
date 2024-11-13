# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import logging
from typing import (
    Optional,
)

import torch

from deepmd.common import (
    j_loader,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
)

log = logging.getLogger(__name__)


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: Optional[str] = None,
) -> None:
    saved_model = torch.jit.load(input_file, map_location="cpu")
    model_def_script = json.loads(saved_model.model_def_script)
    model = get_model(model_def_script)
    model.load_state_dict(saved_model.state_dict())

    if model.get_min_nbor_dist() is None:
        log.info(
            "Minimal neighbor distance is not saved in the model, compute it from the training data."
        )
        if training_script is None:
            raise ValueError(
                "The model does not have a minimum neighbor distance, "
                "so the training script and data must be provided "
                "(via -t,--training-script)."
            )

        jdata = j_loader(training_script)
        jdata = update_deepmd_input(jdata)

        type_map = jdata["model"].get("type_map", None)
        train_data = get_data(
            jdata["training"]["training_data"],
            0,  # not used
            type_map,
            None,
        )
        update_sel = UpdateSel()
        t_min_nbor_dist = update_sel.get_min_nbor_dist(
            train_data,
        )
        model.min_nbor_dist = torch.tensor(
            t_min_nbor_dist,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )

    model.enable_compression(
        extrapolate,
        stride,
        stride * 10,
        check_frequency,
    )

    model = torch.jit.script(model)
    torch.jit.save(model, output)
