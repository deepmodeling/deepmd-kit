# SPDX-License-Identifier: LGPL-3.0-or-later
"""Outer TensorFlow call wrapper for the JAX/jax2tf SavedModel.

The wrapper builds PBC ghosts, neighbor lists, and output communication around
the lower JAX model. It deliberately uses the graph-safe helpers in this
package instead of the TF2 eager helpers, because this code is traced by
``tf.saved_model.save`` and must keep tensor-shape branches convertible by
AutoGraph before it invokes the jax2tf-converted model body.
"""

from collections.abc import (
    Callable,
)

import tensorflow as tf

from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.jax.jax2tf.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.jax.jax2tf.region import (
    normalize_coord,
)
from deepmd.jax.jax2tf.transform_output import (
    communicate_extended_output,
)

CallLower = (
    Callable[
        [
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
        ],
        dict[str, tf.Tensor],
    ]
    | Callable[
        [
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
        ],
        dict[str, tf.Tensor],
    ]
)


def model_call_from_call_lower(
    *,  # enforce keyword-only arguments
    call_lower: CallLower,
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: tf.Tensor,
    atype: tf.Tensor,
    box: tf.Tensor,
    fparam: tf.Tensor,
    aparam: tf.Tensor,
    charge_spin: tf.Tensor | None = None,
    do_atomic_virial: bool = False,
) -> dict[str, tf.Tensor]:
    """Return model prediction from lower interface."""
    atype_shape = tf.shape(atype)
    nframes, nloc = atype_shape[0], atype_shape[1]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    if tf.shape(bb)[-1] != 0:
        coord_normalized = normalize_coord(
            tf.reshape(cc, [nframes, nloc, 3]),
            tf.reshape(bb, [nframes, 3, 3]),
        )
    else:
        coord_normalized = cc
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, bb, rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        # types will be distinguished in the lower interface, so it doesn't
        # need to be distinguished here
        distinguish_types=False,
    )
    extended_coord = tf.reshape(extended_coord, [nframes, -1, 3])
    call_lower_kwargs = {
        "fparam": fp,
        "aparam": ap,
    }
    if charge_spin is not None:
        call_lower_kwargs["charge_spin"] = charge_spin
    model_predict_lower = call_lower(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        **call_lower_kwargs,
    )
    model_predict = communicate_extended_output(
        model_predict_lower,
        model_output_def,
        mapping,
        do_atomic_virial=do_atomic_virial,
    )
    return model_predict
