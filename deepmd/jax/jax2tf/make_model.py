# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
)

import tensorflow as tf
import tensorflow.experimental.numpy as tnp

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
            tnp.ndarray,
            tnp.ndarray,
            tnp.ndarray,
            tnp.ndarray,
            tnp.ndarray,
            bool,
        ],
        dict[str, tnp.ndarray],
    ],
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: tnp.ndarray,
    atype: tnp.ndarray,
    box: tnp.ndarray,
    fparam: tnp.ndarray,
    aparam: tnp.ndarray,
    do_atomic_virial: bool = False,
):
    """Return model prediction from lower interface.

    Parameters
    ----------
    coord
        The coordinates of the atoms.
        shape: nf x (nloc x 3)
    atype
        The type of atoms. shape: nf x nloc
    box
        The simulation box. shape: nf x 9
    fparam
        frame parameter. nf x ndf
    aparam
        atomic parameter. nf x nloc x nda
    do_atomic_virial
        If calculate the atomic virial.

    Returns
    -------
    ret_dict
        The result dict of type dict[str,tnp.ndarray].
        The keys are defined by the `ModelOutputDef`.

    """
    atype_shape = tf.shape(atype)
    nframes, nloc = atype_shape[0], atype_shape[1]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    if tf.shape(bb)[-1] != 0:
        coord_normalized = normalize_coord(
            cc.reshape(nframes, nloc, 3),
            bb.reshape(nframes, 3, 3),
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
        # types will be distinguished in the lower interface,
        # so it doesn't need to be distinguished here
        distinguish_types=False,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)
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
