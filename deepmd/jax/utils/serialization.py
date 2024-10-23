# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

import orbax.checkpoint as ocp

from deepmd.jax.env import (
    jax,
    nnx,
)
from deepmd.jax.model.model import (
    BaseModel,
    get_model,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize the dictionary to a model file.

    Parameters
    ----------
    model_file : str
        The model file to be saved.
    data : dict
        The dictionary to be deserialized.
    """
    if model_file.endswith(".jax"):
        model = BaseModel.deserialize(data["model"])
        model_def_script = data["model_def_script"]
        state = nnx.state(model, ArrayAPIParam)
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    model_def_script=ocp.args.JsonSave(model_def_script),
                ),
            )
    else:
        raise ValueError("JAX backend only supports converting .jax directory")


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
    if model_file.endswith(".jax"):
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            data = checkpointer.restore(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardRestore(),
                    model_def_script=ocp.args.JsonRestore(),
                ),
            )
        state = data.state
        model_def_script = data.model_def_script
        model = get_model(model_def_script)
        nnx.update(model, state)
        model_dict = model.serialize()
        data = {
            "backend": "JAX",
            "jax_version": jax.__version__,
            "model": model_dict,
            "model_def_script": model_def_script,
            "@variables": {},
        }
        return data
    else:
        raise ValueError("JAX backend only supports converting .jax directory")
