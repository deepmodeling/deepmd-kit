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
from typing import (
    TYPE_CHECKING,
)

import tensorflow as tf

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

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


def model_call_from_call_lower(
    *,  # enforce keyword-only arguments
    call_lower: Callable[
        [
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            bool,
        ],
        dict[str, tf.Tensor],
    ],
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: tf.Tensor,
    atype: tf.Tensor,
    box: tf.Tensor,
    fparam: tf.Tensor,
    aparam: tf.Tensor,
    do_atomic_virial: bool = False,
    pair_excl: "PairExcludeMask | None" = None,
) -> dict[str, tf.Tensor]:
    """Return model prediction from lower interface.

    ``pair_excl`` is the model-level pair-type exclusion mask. Exclusion is a
    nlist-BUILD transform (decision #18/A4): it is folded into the nlist here,
    in the traced TF wrapper, because the lower JAX model consumes a
    pre-excluded nlist and never re-applies it.
    """
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
    if pair_excl is not None and len(pair_excl.get_exclude_types()) > 0:
        # Reuse the canonical dpmodel nlist-BUILD transform (decision #18/A4)
        # via the vendored ``ndtensorflow`` array-API namespace -- the same way
        # the TF2 backend (``deepmd/tf2``) runs dpmodel array-API code on
        # TensorFlow. Unlike the neighbor-list *build* (see the docstring of
        # ``jax2tf/nlist.py``), the exclusion has no data-dependent Python
        # control flow: its only branch is on the static ``exclude_types``
        # config, so it traces cleanly under SavedModel export and does not
        # need a hand-written TF twin.
        from deepmd._vendors import (
            ndtensorflow as ndtf,
        )
        from deepmd.dpmodel.utils.nlist import (
            apply_pair_exclusion_nlist,
        )

        nlist = apply_pair_exclusion_nlist(
            ndtf.asarray(nlist), ndtf.asarray(extended_atype), pair_excl
        ).unwrap()
    extended_coord = tf.reshape(extended_coord, [nframes, -1, 3])
    model_predict_lower = call_lower(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fparam=fp,
        aparam=ap,
    )
    model_predict = communicate_extended_output(
        model_predict_lower,
        model_output_def,
        mapping,
        do_atomic_virial=do_atomic_virial,
    )
    return model_predict
