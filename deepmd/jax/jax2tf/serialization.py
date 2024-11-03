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

            return jax2tf.convert(
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
            )

        # Save a function that can take scalar inputs.
        # We need to explicit set the function name, so C++ can find it.
        @tf.function(
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
        def call_lower_without_atomic_virial(
            coord, atype, nlist, mapping, fparam, aparam
        ):
            return exported_whether_do_atomic_virial(do_atomic_virial=False)(
                coord, atype, nlist, mapping, fparam, aparam
            )

        tf_model.call_lower = call_lower_without_atomic_virial

        @tf.function(
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
        def call_lower_with_atomic_virial(coord, atype, nlist, mapping, fparam, aparam):
            return exported_whether_do_atomic_virial(do_atomic_virial=True)(
                coord, atype, nlist, mapping, fparam, aparam
            )

        tf_model.call_lower_atomic_virial = call_lower_with_atomic_virial

        # set functions to export other attributes
        @tf.function
        def get_type_map():
            return tf.constant(model.get_type_map(), dtype=tf.string)

        tf_model.get_type_map = get_type_map

        @tf.function
        def get_rcut():
            return tf.constant(model.get_rcut(), dtype=tf.double)

        tf_model.get_rcut = get_rcut

        @tf.function
        def get_dim_fparam():
            return tf.constant(model.get_dim_fparam(), dtype=tf.int64)

        tf_model.get_dim_fparam = get_dim_fparam

        @tf.function
        def get_dim_aparam():
            return tf.constant(model.get_dim_aparam(), dtype=tf.int64)

        tf_model.get_dim_aparam = get_dim_aparam

        @tf.function
        def get_sel_type():
            return tf.constant(model.get_sel_type(), dtype=tf.int64)

        tf_model.get_sel_type = get_sel_type

        @tf.function
        def is_aparam_nall():
            return tf.constant(model.is_aparam_nall(), dtype=tf.bool)

        tf_model.is_aparam_nall = is_aparam_nall

        @tf.function
        def model_output_type():
            return tf.constant(model.model_output_type(), dtype=tf.string)

        tf_model.model_output_type = model_output_type

        @tf.function
        def mixed_types():
            return tf.constant(model.mixed_types(), dtype=tf.bool)

        tf_model.mixed_types = mixed_types

        if model.get_min_nbor_dist() is not None:

            @tf.function
            def get_min_nbor_dist():
                return tf.constant(model.get_min_nbor_dist(), dtype=tf.double)

            tf_model.get_min_nbor_dist = get_min_nbor_dist

        @tf.function
        def get_sel():
            return tf.constant(model.get_sel(), dtype=tf.int64)

        tf_model.get_sel = get_sel

        @tf.function
        def get_model_def_script():
            return tf.constant(
                json.dumps(model_def_script, separators=(",", ":")), dtype=tf.string
            )

        tf_model.get_model_def_script = get_model_def_script
        tf.saved_model.save(
            tf_model,
            model_file,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
        )
