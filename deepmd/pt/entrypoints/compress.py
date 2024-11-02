# SPDX-License-Identifier: LGPL-3.0-or-later
import json

import torch

from deepmd.pt.model.model import (
    get_model,
)

def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
):
    saved_model = torch.jit.load(input_file, map_location="cpu")
    model_def_script = json.loads(saved_model.model_def_script)
    model = get_model(model_def_script)
    model.load_state_dict(saved_model.state_dict())

    model.enable_compression(
        extrapolate,
        stride,
        stride * 10,
        check_frequency,
    )

    model = torch.jit.script(model)
    torch.jit.save(model, output)
