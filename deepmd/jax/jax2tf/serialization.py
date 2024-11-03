# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf
from jax.experimental import (
    jax2tf,
)

from deepmd.jax.model.base_model import (
    BaseModel,
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
    if model_file.endswith(".savedmodel"):
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
                    f"(nf, {model.get_dim_fparam()})",
                    f"(nf, nloc, {model.get_dim_aparam()})",
                ],
            ),
            autograph=False,
            input_signature=[
                tf.TensorSpec([None, None, 3], tf.float64),
                tf.TensorSpec([None, None], tf.int64),
                tf.TensorSpec([None, None, model.get_nnei()], tf.int64),
                tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
                tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
            ],
        )
        my_model.model_def_script = model_def_script
        tf.saved_model.save(
            my_model,
            model_file,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
        )
