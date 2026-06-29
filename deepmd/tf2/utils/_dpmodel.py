# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list as dpmodel_build_neighbor_list,
)
from deepmd.dpmodel.utils.nlist import (
    extend_coord_with_ghosts as dpmodel_extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.nlist import format_nlist as dpmodel_format_nlist
from deepmd.dpmodel.utils.region import inter2phys as dpmodel_inter2phys
from deepmd.dpmodel.utils.region import normalize_coord as dpmodel_normalize_coord
from deepmd.dpmodel.utils.region import to_face_distance as dpmodel_to_face_distance
from deepmd.tf2.common import (
    to_tensorflow_array,
    to_tf_tensor,
)
from deepmd.tf2.env import (
    Array,
)


def build_neighbor_list(
    coord: Any,
    atype: Any,
    nloc: int,
    rcut: float,
    sel: int | list[int],
    distinguish_types: bool = True,
) -> Array:
    ret = to_tf_tensor(
        dpmodel_build_neighbor_list(
            to_tensorflow_array(coord),
            to_tensorflow_array(atype),
            nloc,
            rcut,
            sel,
            distinguish_types=distinguish_types,
        )
    )
    ret.set_shape([None, None, sel if isinstance(sel, int) else sum(sel)])
    return to_tensorflow_array(ret)


def extend_coord_with_ghosts(
    coord: Any,
    atype: Any,
    cell: Any | None,
    rcut: float,
) -> tuple[Array, Array, Array]:
    extended_coord, extended_atype, mapping = dpmodel_extend_coord_with_ghosts(
        to_tensorflow_array(coord),
        to_tensorflow_array(atype),
        None if cell is None else to_tensorflow_array(cell),
        rcut,
    )
    return (
        to_tensorflow_array(extended_coord),
        to_tensorflow_array(extended_atype),
        to_tensorflow_array(mapping),
    )


def format_nlist(
    extended_coord: Any,
    nlist: Any,
    nsel: int,
    rcut: float,
) -> Array:
    return to_tensorflow_array(
        dpmodel_format_nlist(
            to_tensorflow_array(extended_coord),
            to_tensorflow_array(nlist),
            nsel,
            rcut,
        )
    )


def inter2phys(coord: Any, cell: Any) -> Array:
    return to_tensorflow_array(
        dpmodel_inter2phys(to_tensorflow_array(coord), to_tensorflow_array(cell))
    )


def normalize_coord(coord: Any, cell: Any) -> Array:
    return to_tensorflow_array(
        dpmodel_normalize_coord(to_tensorflow_array(coord), to_tensorflow_array(cell))
    )


def to_face_distance(cell: Any) -> Array:
    return to_tensorflow_array(dpmodel_to_face_distance(to_tensorflow_array(cell)))


__all__ = [
    "build_neighbor_list",
    "extend_coord_with_ghosts",
    "format_nlist",
    "inter2phys",
    "normalize_coord",
    "to_face_distance",
]
