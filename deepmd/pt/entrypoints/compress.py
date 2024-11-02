# SPDX-License-Identifier: LGPL-3.0-or-later
import json

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
):
    if input_file.endswith(".pth"):
        saved_model = torch.jit.load(input_file, map_location="cpu")
        model_def_script = json.loads(saved_model.model_def_script)
        model = get_model(model_def_script)
        model.load_state_dict(saved_model.state_dict())
    elif input_file.endswith(".pt"):
        state_dict = torch.load(input_file, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_def_script = state_dict["_extra_state"]["model_params"]
        model = get_model(model_def_script)
        modelwrapper = ModelWrapper(model)
        modelwrapper.load_state_dict(state_dict)
        model = modelwrapper.model["Default"]
    else:
        raise ValueError("PyTorch backend only supports converting .pth or .pt file")

    model.enable_compression(
        extrapolate,
        stride,
        stride * 10,
        check_frequency,
    )

    model = torch.jit.script(model)
    torch.jit.save(model, output)
