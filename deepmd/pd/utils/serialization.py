# SPDX-License-Identifier: LGPL-3.0-or-later
import json

import paddle

from deepmd.pd.model.model.model import (
    BaseModel,
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
    raise NotImplementedError("Paddle do not support jit.export yet.")


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    model_file : str
        The model file to be saved.
    data : dict
        The dictionary to be deserialized.
    """
    if not model_file.endswith(".json"):
        raise ValueError("Paddle backend only supports converting .json file")
    model = BaseModel.deserialize(data["model"])
    # JIT will happy in this way...
    model.model_def_script = json.dumps(data["model_def_script"])
    if "min_nbor_dist" in data.get("@variables", {}):
        model.min_nbor_dist = float(data["@variables"]["min_nbor_dist"])
    # model = paddle.jit.to_static(model)
    paddle.set_flags(
        {
            "FLAGS_save_cf_stack_op": 1,
            "FLAGS_prim_enable_dynamic": 1,
            "FLAGS_enable_pir_api": 1,
        }
    )
    paddle.jit.save(
        model,
        model_file.split(".json")[0],
    )
