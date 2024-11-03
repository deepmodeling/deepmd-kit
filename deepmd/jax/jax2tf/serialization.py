# SPDX-License-Identifier: LGPL-3.0-or-later
import json

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

        tf_model = tf.Module()

        def exported_whether_do_atomic_virial(do_atomic_virial):
            def call_lower_with_fixed_do_atomic_virial(
                coord, atype, nlist, mapping, fparam, aparam
            ):
                return call_lower(
                    coord,
                    atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    do_atomic_virial=do_atomic_virial,
                )

            return tf.function(
                jax2tf.convert(
                    call_lower_with_fixed_do_atomic_virial,
                    polymorphic_shapes=[
                        "(nf, nloc + nghost, 3)",
                        "(nf, nloc + nghost)",
                        f"(nf, nloc, {model.get_nnei()})",
                        "(nf, nloc + nghost)",
                        f"(nf, {model.get_dim_fparam()})",
                        f"(nf, nloc, {model.get_dim_aparam()})",
                    ],
                    with_gradient=True,
                ),
                autograph=False,
                input_signature=[
                    tf.TensorSpec([None, None, 3], tf.float64),
                    tf.TensorSpec([None, None], tf.int32),
                    tf.TensorSpec([None, None, model.get_nnei()], tf.int64),
                    tf.TensorSpec([None, None], tf.int64),
                    tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
                    tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
                ],
            )

        # Save a function that can take scalar inputs.
        tf_model.call_lower = exported_whether_do_atomic_virial(do_atomic_virial=False)
        tf_model.call_lower_atomic_virial = exported_whether_do_atomic_virial(
            do_atomic_virial=True
        )
        # set functions to export other attributes
        tf_model.get_type_map = tf.function(
            lambda: tf.constant(model.get_type_map(), dtype=tf.string)
        )
        tf_model.get_rcut = tf.function(
            lambda: tf.constant(model.get_rcut(), dtype=tf.double)
        )
        tf_model.get_dim_fparam = tf.function(
            lambda: tf.constant(model.get_dim_fparam(), dtype=tf.int64)
        )
        tf_model.get_dim_aparam = tf.function(
            lambda: tf.constant(model.get_dim_aparam(), dtype=tf.int64)
        )
        tf_model.get_sel_type = tf.function(
            lambda: tf.constant(model.get_sel_type(), dtype=tf.int64)
        )
        tf_model.is_aparam_nall = tf.function(
            lambda: tf.constant(model.is_aparam_nall(), dtype=tf.bool)
        )
        tf_model.model_output_type = tf.function(
            lambda: tf.constant(model.model_output_type(), dtype=tf.string)
        )
        tf_model.mixed_types = tf.function(
            lambda: tf.constant(model.mixed_types(), dtype=tf.bool)
        )
        if model.get_min_nbor_dist() is not None:
            tf_model.get_min_nbor_dist = tf.function(
                lambda: tf.constant(model.get_min_nbor_dist(), dtype=tf.double)
            )
        tf_model.get_sel = tf.function(
            lambda: tf.constant(model.get_sel(), dtype=tf.int64)
        )
        tf_model.get_model_def_script = tf.function(
            lambda: tf.constant(
                json.dumps(model_def_script, separators=(",", ":")), dtype=tf.string
            )
        )
        tf.saved_model.save(
            tf_model,
            model_file,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
        )
