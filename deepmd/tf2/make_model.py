# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import tensorflow as tf

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.tf2.common import (
    to_tensorflow_array,
    to_tf_tensor,
    wrap_value,
)
from deepmd.tf2.env import (
    xp,
)
from deepmd.tf2.utils._dpmodel import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    normalize_coord,
)


def _unwrap_tuple(values: tuple[Array, ...]) -> tuple[tf.Tensor, ...]:
    return tuple(to_tf_tensor(value) for value in values)


def _box_has_pbc(box: Array | None) -> bool | None:
    if box is None:
        return False
    last_dim = box.shape[-1]
    return (last_dim != 0) if isinstance(last_dim, int) else None


def model_call_from_call_lower(
    *,  # enforce keyword-only arguments
    call_lower: Callable[..., dict[str, Any]],
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: Array,
    atype: Array,
    box: Array | None,
    fparam: Array | None,
    aparam: Array | None,
    do_atomic_virial: bool = False,
) -> dict[str, Array]:
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
        The result dict of type dict[str, Array].
        The keys are defined by the `ModelOutputDef`.

    """
    cc = to_tensorflow_array(coord)
    atype = to_tensorflow_array(atype)
    bb = to_tensorflow_array(box)
    fp = to_tensorflow_array(fparam)
    ap = to_tensorflow_array(aparam)
    del coord, box, fparam, aparam
    nframes, nloc = atype.shape[:2]

    def with_pbc() -> tuple[Array, Array, Array]:
        assert bb is not None
        coord_normalized = normalize_coord(
            xp.reshape(cc, (nframes, nloc, 3)),
            xp.reshape(bb, (nframes, 3, 3)),
        )
        return extend_coord_with_ghosts(coord_normalized, atype, bb, rcut)

    def no_pbc() -> tuple[Array, Array, Array]:
        return extend_coord_with_ghosts(cc, atype, None, rcut)

    has_pbc = _box_has_pbc(bb)
    if has_pbc is True:
        extended_coord, extended_atype, mapping = with_pbc()
    elif has_pbc is False:
        extended_coord, extended_atype, mapping = no_pbc()
    else:
        assert bb is not None
        extended_coord_tensor, extended_atype_tensor, mapping_tensor = tf.cond(
            tf.shape(to_tf_tensor(bb))[-1] != 0,
            lambda: _unwrap_tuple(with_pbc()),
            lambda: _unwrap_tuple(no_pbc()),
        )
        extended_coord = to_tensorflow_array(extended_coord_tensor)
        extended_atype = to_tensorflow_array(extended_atype_tensor)
        mapping = to_tensorflow_array(mapping_tensor)
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
    extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
    model_predict_lower = wrap_value(
        call_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fp,
            aparam=ap,
        )
    )
    model_predict = communicate_extended_output(
        model_predict_lower,
        model_output_def,
        mapping,
        do_atomic_virial=do_atomic_virial,
    )
    return model_predict
