# SPDX-License-Identifier: LGPL-3.0-or-later

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
    model: paddle.nn.Layer = BaseModel.deserialize(data["model"])
    model.eval()
    # JIT will happy in this way...
    if "min_nbor_dist" in data.get("@variables", {}):
        model.register_buffer(
            "buffer_min_nbor_dist",
            paddle.to_tensor(
                float(data["@variables"]["min_nbor_dist"]),
            ),
        )
    paddle.set_flags(
        {
            "FLAGS_save_cf_stack_op": 1,
            "FLAGS_prim_enable_dynamic": 1,
            "FLAGS_enable_pir_api": 1,
        }
    )
    from paddle.static import (
        InputSpec,
    )

    jit_model = paddle.jit.to_static(
        model,
        full_graph=True,
        input_spec=[
            InputSpec([1, -1, 3], dtype="float64", name="coord"),
            InputSpec([1, -1], dtype="int64", name="atype"),
            InputSpec([1, 9], dtype="float64", name="box"),
        ],
    )
    paddle.jit.save(
        jit_model,
        model_file.split(".json")[0],
    )
