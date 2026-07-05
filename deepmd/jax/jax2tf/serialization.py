# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX/jax2tf SavedModel export.

The ``.savedmodel`` suffix is the JAX SavedModel artifact used by the JAX C++
inference path. It is intentionally different from the TF2 eager
``.savedmodeltf`` artifact: the model body below must pass through
``jax2tf.convert`` so TensorFlow stores XlaCallModule nodes. Do not replace this
module with the TF2 SavedModel exporter unless the file suffix and C++ loader
contract are changed together.
"""

import json
from collections.abc import (
    Callable,
)

import tensorflow as tf
from jax.experimental import (
    jax2tf,
)

from deepmd.jax.jax2tf.format_nlist import (
    format_nlist,
)
from deepmd.jax.jax2tf.make_model import (
    model_call_from_call_lower,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.utils.serialization import (
    _set_model_min_nbor_dist_from_data,
)


def deserialize_to_file(model_file: str, data: dict) -> None:
    """Deserialize the dictionary to a JAX/jax2tf SavedModel."""
    if model_file.endswith(".savedmodel"):
        model = BaseModel.deserialize(data["model"])
        _set_model_min_nbor_dist_from_data(model, data)
        model_def_script = data["model_def_script"]
        call_lower = model.call_common_lower
        dim_chg_spin = model.get_dim_chg_spin()
        has_chg_spin = dim_chg_spin > 0

        tf_model = tf.Module()

        def lower_input_signature() -> list[tf.TensorSpec]:
            signature = [
                tf.TensorSpec([None, None, 3], tf.float64),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, None, None], tf.int64),
                tf.TensorSpec([None, None], tf.int64),
                tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
                tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
            ]
            if has_chg_spin:
                signature.append(tf.TensorSpec([None, dim_chg_spin], tf.float64))
            return signature

        def call_input_signature() -> list[tf.TensorSpec]:
            signature = [
                tf.TensorSpec([None, None, 3], tf.float64),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, None, None], tf.float64),
                tf.TensorSpec([None, model.get_dim_fparam()], tf.float64),
                tf.TensorSpec([None, None, model.get_dim_aparam()], tf.float64),
            ]
            if has_chg_spin:
                signature.append(tf.TensorSpec([None, dim_chg_spin], tf.float64))
            return signature

        def lower_args(
            coord: tf.Tensor,
            atype: tf.Tensor,
            nlist: tf.Tensor,
            mapping: tf.Tensor,
            fparam: tf.Tensor,
            aparam: tf.Tensor,
            charge_spin: tf.Tensor | None,
        ) -> tuple[tf.Tensor, ...]:
            args = (coord, atype, nlist, mapping, fparam, aparam)
            if has_chg_spin:
                assert charge_spin is not None
                args = (*args, charge_spin)
            return args

        def exported_whether_do_atomic_virial(
            do_atomic_virial: bool, has_ghost_atoms: bool
        ) -> Callable:
            if has_chg_spin:

                def call_lower_with_fixed_do_atomic_virial(
                    coord: tf.Tensor,
                    atype: tf.Tensor,
                    nlist: tf.Tensor,
                    mapping: tf.Tensor,
                    fparam: tf.Tensor,
                    aparam: tf.Tensor,
                    charge_spin: tf.Tensor,
                ) -> dict[str, tf.Tensor]:
                    return call_lower(
                        coord,
                        atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                        do_atomic_virial=do_atomic_virial,
                        charge_spin=charge_spin,
                    )

            else:

                def call_lower_with_fixed_do_atomic_virial(
                    coord: tf.Tensor,
                    atype: tf.Tensor,
                    nlist: tf.Tensor,
                    mapping: tf.Tensor,
                    fparam: tf.Tensor,
                    aparam: tf.Tensor,
                ) -> dict[str, tf.Tensor]:
                    return call_lower(
                        coord,
                        atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                        do_atomic_virial=do_atomic_virial,
                    )

            # nghost >= 1 is assumed if there is ghost atoms. Other workarounds
            # do not work, such as nall; nloc + nghost - 1.
            if has_ghost_atoms:
                nghost = "nghost"
            else:
                nghost = "0"
            # The converted function is the part that carries the JAX model
            # semantics into TensorFlow. Its SavedModel graph is expected to
            # contain XlaCallModule ops; a graph made only of ordinary TF ops
            # means this path has accidentally fallen back to the TF2 exporter.
            polymorphic_shapes = [
                f"(nf, nloc + {nghost}, 3)",
                f"(nf, nloc + {nghost})",
                f"(nf, nloc, {model.get_nnei()})",
                f"(nf, nloc + {nghost})",
                f"(nf, {model.get_dim_fparam()})",
                f"(nf, nloc, {model.get_dim_aparam()})",
            ]
            if has_chg_spin:
                polymorphic_shapes.append(f"(nf, {dim_chg_spin})")
            return jax2tf.convert(
                call_lower_with_fixed_do_atomic_virial,
                polymorphic_shapes=polymorphic_shapes,
                with_gradient=True,
            )

        def dispatch_call_lower(
            do_atomic_virial: bool,
            coord: tf.Tensor,
            atype: tf.Tensor,
            nlist: tf.Tensor,
            mapping: tf.Tensor,
            fparam: tf.Tensor,
            aparam: tf.Tensor,
            charge_spin: tf.Tensor | None = None,
        ) -> dict[str, tf.Tensor]:
            nlist = format_nlist(coord, nlist, model.get_nnei(), model.get_rcut())
            args = lower_args(coord, atype, nlist, mapping, fparam, aparam, charge_spin)
            return tf.cond(
                tf.shape(coord)[1] == tf.shape(nlist)[1],
                lambda: exported_whether_do_atomic_virial(
                    do_atomic_virial=do_atomic_virial, has_ghost_atoms=False
                )(*args),
                lambda: exported_whether_do_atomic_virial(
                    do_atomic_virial=do_atomic_virial, has_ghost_atoms=True
                )(*args),
            )

        if has_chg_spin:

            @tf.function(autograph=False, input_signature=lower_input_signature())
            def call_lower_without_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                nlist: tf.Tensor,
                mapping: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
                charge_spin: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return dispatch_call_lower(
                    False, coord, atype, nlist, mapping, fparam, aparam, charge_spin
                )

        else:

            @tf.function(autograph=False, input_signature=lower_input_signature())
            def call_lower_without_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                nlist: tf.Tensor,
                mapping: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return dispatch_call_lower(
                    False, coord, atype, nlist, mapping, fparam, aparam
                )

        tf_model.call_lower = call_lower_without_atomic_virial

        if has_chg_spin:

            @tf.function(autograph=False, input_signature=lower_input_signature())
            def call_lower_with_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                nlist: tf.Tensor,
                mapping: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
                charge_spin: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return dispatch_call_lower(
                    True, coord, atype, nlist, mapping, fparam, aparam, charge_spin
                )

        else:

            @tf.function(autograph=False, input_signature=lower_input_signature())
            def call_lower_with_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                nlist: tf.Tensor,
                mapping: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return dispatch_call_lower(
                    True, coord, atype, nlist, mapping, fparam, aparam
                )

        tf_model.call_lower_atomic_virial = call_lower_with_atomic_virial

        def make_call_whether_do_atomic_virial(do_atomic_virial: bool) -> Callable:
            if do_atomic_virial:
                call_lower = call_lower_with_atomic_virial
            else:
                call_lower = call_lower_without_atomic_virial

            def call(
                coord: tf.Tensor,
                atype: tf.Tensor,
                box: tf.Tensor | None = None,
                fparam: tf.Tensor | None = None,
                aparam: tf.Tensor | None = None,
                charge_spin: tf.Tensor | None = None,
            ) -> dict[str, tf.Tensor]:
                return model_call_from_call_lower(
                    call_lower=call_lower,
                    rcut=model.get_rcut(),
                    sel=model.get_sel(),
                    mixed_types=model.mixed_types(),
                    model_output_def=model.model_output_def(),
                    coord=coord,
                    atype=atype,
                    box=box,
                    fparam=fparam,
                    aparam=aparam,
                    charge_spin=charge_spin,
                    do_atomic_virial=do_atomic_virial,
                )

            return call

        if has_chg_spin:

            @tf.function(autograph=True, input_signature=call_input_signature())
            def call_with_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                box: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
                charge_spin: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return make_call_whether_do_atomic_virial(do_atomic_virial=True)(
                    coord, atype, box, fparam, aparam, charge_spin
                )

        else:

            @tf.function(autograph=True, input_signature=call_input_signature())
            def call_with_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                box: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return make_call_whether_do_atomic_virial(do_atomic_virial=True)(
                    coord, atype, box, fparam, aparam
                )

        tf_model.call_atomic_virial = call_with_atomic_virial

        if has_chg_spin:

            @tf.function(autograph=True, input_signature=call_input_signature())
            def call_without_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                box: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
                charge_spin: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return make_call_whether_do_atomic_virial(do_atomic_virial=False)(
                    coord, atype, box, fparam, aparam, charge_spin
                )

        else:

            @tf.function(autograph=True, input_signature=call_input_signature())
            def call_without_atomic_virial(
                coord: tf.Tensor,
                atype: tf.Tensor,
                box: tf.Tensor,
                fparam: tf.Tensor,
                aparam: tf.Tensor,
            ) -> dict[str, tf.Tensor]:
                return make_call_whether_do_atomic_virial(do_atomic_virial=False)(
                    coord, atype, box, fparam, aparam
                )

        tf_model.call = call_without_atomic_virial

        @tf.function
        def get_type_map() -> tf.Tensor:
            return tf.constant(model.get_type_map(), dtype=tf.string)

        tf_model.get_type_map = get_type_map

        @tf.function
        def get_rcut() -> tf.Tensor:
            return tf.constant(model.get_rcut(), dtype=tf.double)

        tf_model.get_rcut = get_rcut

        @tf.function
        def get_dim_fparam() -> tf.Tensor:
            return tf.constant(model.get_dim_fparam(), dtype=tf.int64)

        tf_model.get_dim_fparam = get_dim_fparam

        @tf.function
        def get_dim_aparam() -> tf.Tensor:
            return tf.constant(model.get_dim_aparam(), dtype=tf.int64)

        tf_model.get_dim_aparam = get_dim_aparam

        @tf.function
        def get_sel_type() -> tf.Tensor:
            return tf.constant(model.get_sel_type(), dtype=tf.int64)

        tf_model.get_sel_type = get_sel_type

        @tf.function
        def is_aparam_nall() -> tf.Tensor:
            return tf.constant(model.is_aparam_nall(), dtype=tf.bool)

        tf_model.is_aparam_nall = is_aparam_nall

        @tf.function
        def model_output_type() -> tf.Tensor:
            return tf.constant(model.model_output_type(), dtype=tf.string)

        tf_model.model_output_type = model_output_type

        @tf.function
        def mixed_types() -> tf.Tensor:
            return tf.constant(model.mixed_types(), dtype=tf.bool)

        tf_model.mixed_types = mixed_types

        if model.get_min_nbor_dist() is not None:

            @tf.function
            def get_min_nbor_dist() -> tf.Tensor:
                return tf.constant(model.get_min_nbor_dist(), dtype=tf.double)

            tf_model.get_min_nbor_dist = get_min_nbor_dist

        @tf.function
        def get_sel() -> tf.Tensor:
            return tf.constant(model.get_sel(), dtype=tf.int64)

        tf_model.get_sel = get_sel

        @tf.function
        def get_model_def_script() -> tf.Tensor:
            return tf.constant(
                json.dumps(model_def_script, separators=(",", ":")), dtype=tf.string
            )

        tf_model.get_model_def_script = get_model_def_script

        @tf.function
        def has_message_passing() -> tf.Tensor:
            return tf.constant(model.has_message_passing(), dtype=tf.bool)

        tf_model.has_message_passing = has_message_passing
        tf_model.do_message_passing = has_message_passing

        @tf.function
        def has_default_fparam() -> tf.Tensor:
            return tf.constant(model.has_default_fparam(), dtype=tf.bool)

        tf_model.has_default_fparam = has_default_fparam

        @tf.function
        def get_default_fparam() -> tf.Tensor:
            default_fparam = model.get_default_fparam()
            if default_fparam is None:
                return tf.constant([], dtype=tf.double)
            return tf.constant(default_fparam, dtype=tf.double)

        tf_model.get_default_fparam = get_default_fparam

        @tf.function
        def has_chg_spin_ebd() -> tf.Tensor:
            return tf.constant(model.has_chg_spin_ebd(), dtype=tf.bool)

        tf_model.has_chg_spin_ebd = has_chg_spin_ebd

        @tf.function
        def get_dim_chg_spin() -> tf.Tensor:
            return tf.constant(dim_chg_spin, dtype=tf.int64)

        tf_model.get_dim_chg_spin = get_dim_chg_spin

        @tf.function
        def has_default_chg_spin() -> tf.Tensor:
            return tf.constant(model.has_default_chg_spin(), dtype=tf.bool)

        tf_model.has_default_chg_spin = has_default_chg_spin

        @tf.function
        def get_default_chg_spin() -> tf.Tensor:
            default_chg_spin = model.get_default_chg_spin()
            if default_chg_spin is None:
                return tf.constant([], dtype=tf.double)
            return tf.constant(default_chg_spin, dtype=tf.double)

        tf_model.get_default_chg_spin = get_default_chg_spin

        tf.saved_model.save(
            tf_model,
            model_file,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
        )
