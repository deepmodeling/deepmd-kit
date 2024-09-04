# SPDX-License-Identifier: LGPL-3.0-or-later
import json

import paddle

from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.model.model.model import (
    BaseModel,
)
from deepmd.pd.train.wrapper import (
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
        The serialized model data.
    """
    if model_file.endswith(".pdparams"):
        saved_model = paddle.jit.load(model_file)
        model_def_script = json.loads(saved_model.model_def_script)
        model = get_model(model_def_script)
        model.set_state_dict(saved_model.state_dict())
    elif model_file.endswith(".pdmodel"):
        state_dict = paddle.load(model_file)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_def_script = state_dict["_extra_state"]["model_params"]
        model = get_model(model_def_script)
        modelwrapper = ModelWrapper(model)
        modelwrapper.set_state_dict(state_dict)
        model = modelwrapper.model["Default"]
    else:
        raise ValueError(
            "Pypaddle backend only supports converting .pdparams or .pd file"
        )

    model_dict = model.serialize()
    data = {
        "backend": "Pypaddle",
        "pt_version": paddle.__version__,
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
    if not model_file.endswith(".pdparams"):
        raise ValueError("Pypaddle backend only supports converting .pdparams file")
    model = BaseModel.deserialize(data["model"])
    # JIT will happy in this way...
    model.model_def_script = json.dumps(data["model_def_script"])
    if "min_nbor_dist" in data.get("@variables", {}):
        model.min_nbor_dist = float(data["@variables"]["min_nbor_dist"])
    model = paddle.jit.to_static(model)
    paddle.jit.save(model, model_file)
