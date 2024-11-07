# SPDX-License-Identifier: LGPL-3.0-or-later
import json

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    env,
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
    if model_file.endswith(".pth"):
        saved_model = torch.jit.load(model_file, map_location="cpu")
        model_def_script = json.loads(saved_model.model_def_script)
        model = get_model(model_def_script)
        model.load_state_dict(saved_model.state_dict())
    elif model_file.endswith(".pt"):
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_def_script = state_dict["_extra_state"]["model_params"]
        model = get_model(model_def_script)
        modelwrapper = ModelWrapper(model)
        modelwrapper.load_state_dict(state_dict)
        model = modelwrapper.model["Default"]
    else:
        raise ValueError("PyTorch backend only supports converting .pth or .pt file")

    model_dict = model.serialize()
    data = {
        "backend": "PyTorch",
        "pt_version": str(torch.__version__),
        "model": model_dict,
        "model_def_script": model_def_script,
        "@variables": {},
    }
    if model.get_min_nbor_dist() is not None:
        data["@variables"]["min_nbor_dist"] = model.get_min_nbor_dist()
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
    model = BaseModel.deserialize(data["model"])
    # JIT will happy in this way...
    model.model_def_script = json.dumps(data["model_def_script"])
    if "min_nbor_dist" in data.get("@variables", {}):
        model.min_nbor_dist = torch.tensor(
            float(data["@variables"]["min_nbor_dist"]),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
    model = torch.jit.script(model)
    torch.jit.save(model, model_file)
