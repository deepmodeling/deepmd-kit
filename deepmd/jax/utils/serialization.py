# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf

from deepmd.jax.env import (
    jax2tf,
    nnx,
)
from deepmd.jax.model.model import (
    BaseModel,
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
    if model_file.endswith(".saved_model"):
        model = BaseModel.deserialize(data["model"])
        model_def_script = data.get("model_def_script", "{}")
        my_model = tf.Module()
        my_model.f = tf.function(
            jax2tf.convert(
                model,
                polymorphic_shapes=[
                    "(b, n, 3)",
                    "(b, n)",
                    "(b, 3, 3)",
                    "(b, f)",
                    "(b, a)",
                    "()",
                ],
            ),
            autograph=False,
            input_signature=[
                tf.TensorSpec([None, None, 3], tf.float64),
                tf.TensorSpec([None, None], tf.int64),
                tf.TensorSpec([None, 3, 3], tf.float64),
                tf.TensorSpec([None, None], tf.float64),
                tf.TensorSpec([None, None], tf.float64),
                tf.TensorSpec([], tf.bool),
            ],
        )
        my_model.model_def_script = model_def_script
        tf.saved_model.save(
            my_model,
            model_file,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
        )
    elif model_file.endswith(".jax"):
        model = BaseModel.deserialize(data["model"])
        state = nnx.state(model, ArrayAPIParam)
        nnx.display(state)
    else:
        raise ValueError("JAX backend only supports converting .pth file")
