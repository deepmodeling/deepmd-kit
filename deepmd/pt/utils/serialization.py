# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)


def serialize_from_file(model_file: str) -> dict:
    """Serialize the model file to a dictionary.

    Parameters
    ----------
    model_file : str
        The model file to be serialized.

    Returns
    -------
    dict
        The serialized model file.
    """
    if model_file.endswith(".pt"):
        state_dict = torch.load(model_file, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        input_param = state_dict["_extra_state"]["model_params"]
        input_param["resuming"] = True
        multi_task = "model_dict" in input_param
        assert not multi_task, "multitask mode currently not supported!"
        model = get_model(input_param).to("cpu")
        model = torch.jit.script(model)
        dp = ModelWrapper(model)
        dp.load_state_dict(state_dict)
    elif model_file.endswith(".pth"):
        model = torch.jit.load(model_file, map_location="cpu")
        dp = ModelWrapper(model)
    else:
        raise ValueError(f"Unsupported model file format: {model_file}")
    model = dp.model["Default"]
    model_dict = model.serialize()
    data = {
        "backend": "PyTorch",
        "model": model_dict,
    }
    return data


def deserialize_to_file(data: dict, model_file: str) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    data : dict
        The dictionary to be deserialized.
    model_file : str
        The model file to be saved.
    """
