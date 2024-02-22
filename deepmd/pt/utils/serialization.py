# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.pt.model.model.dp_model import (
    DPModel,
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
    model = torch.jit.load(model_file, map_location="cpu")
    model_dict = model.serialize()
    data = {
        "backend": "PyTorch",
        "pt_version": torch.__version__,
        "model": model_dict,
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
    # TODO: see #3319
    model = DPModel.deserialize(data)
    model = torch.jit.script(model)
    torch.jit.save(model, model_file)
