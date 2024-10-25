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
        _, state = nnx.split(model)
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state.to_pure_dict()),
                    model_def_script=ocp.args.JsonSave(model_def_script),
                ),
            )
    elif model_file.endswith(".savedmodel"):
        import tensorflow as tf
        from jax.experimental import (
            jax2tf,
        )

        model = BaseModel.deserialize(data["model"])
        model_def_script = data["model_def_script"]
        call_lower = model.call_lower

        my_model = tf.Module()

        # Save a function that can take scalar inputs.
        my_model.call_lower = tf.function(
            jax2tf.convert(
                call_lower,
                polymorphic_shapes=[
                    "(nf, nloc + nghost, 3)",
                    "(nf, nloc + nghost)",
                    f"(nf, nloc, {model.get_nnei()})",
                    "(nf, np)",
                    "(nf, na)",
                ],
            ),
            autograph=False,
            input_signature=[
                tf.TensorSpec([None, None, 3], tf.float64),
                tf.TensorSpec([None, None], tf.int64),
                tf.TensorSpec([None, None, model.get_nnei()], tf.int64),
                tf.TensorSpec([None, None], tf.float64),
                tf.TensorSpec([None, None], tf.float64),
            ],
        )
        my_model.model_def_script = model_def_script
        tf.saved_model.save(
            my_model,
            model_file,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
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

        # convert str "1" to int 1 key
        def convert_str_to_int_key(item: dict):
            for key, value in item.copy().items():
                if isinstance(value, dict):
                    convert_str_to_int_key(value)
                if key.isdigit():
                    item[int(key)] = item.pop(key)

        convert_str_to_int_key(state)

        model_def_script = data.model_def_script
        abstract_model = get_model(model_def_script)
        graphdef, abstract_state = nnx.split(abstract_model)
        abstract_state.replace_by_pure_dict(state)
        model = nnx.merge(graphdef, abstract_state)
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
