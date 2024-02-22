# SPDX-License-Identifier: LGPL-3.0-or-later
import json

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.ener_model import (
    EnergyModel,
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
        The serialized model data.
    """
    if not model_file.endswith(".pth"):
        raise ValueError("PyTorch backend only supports converting .pth file")
    jit_model = torch.jit.load(model_file, map_location="cpu")
    model_param = json.loads(jit_model.model_param)
    model = get_model(model_param)
    model.load_state_dict(jit_model.state_dict())
    model_dict = model.serialize()
    data = {
        "backend": "PyTorch",
        "pt_version": torch.__version__,
        "model": model_dict,
        "model_param": model_param,
        # TODO
        "@variables": {},
    }
    return data


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    model_file : str
        The model file to be saved.
    data : dict
        The dictionary to be deserialized.
    """
    if not model_file.endswith(".pth"):
        raise ValueError("PyTorch backend only supports converting .pth file")
    # TODO: read class type from data; see #3319
    model = EnergyModel.deserialize(data["model"])
    # JIT will happy in this way...
    model.model_param = json.dumps(data["model_param"])
    model = torch.jit.script(model)
    torch.jit.save(model, model_file)
