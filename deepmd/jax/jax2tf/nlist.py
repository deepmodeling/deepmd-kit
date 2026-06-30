# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility wrappers for TensorFlow neighbor-list helpers."""

from typing import (
    Any,
)

import tensorflow as tf

from deepmd.tf2.common import (
    to_tf_tensor,
)
from deepmd.tf2.utils._dpmodel import build_neighbor_list as tf2_build_neighbor_list
from deepmd.tf2.utils._dpmodel import (
    extend_coord_with_ghosts as tf2_extend_coord_with_ghosts,
)

__all__ = [
    "build_neighbor_list",
    "extend_coord_with_ghosts",
]


def build_neighbor_list(
    coord: Any,
    atype: Any,
    nloc: int,
    rcut: float,
    sel: int | list[int],
    distinguish_types: bool = True,
) -> tf.Tensor:
    return to_tf_tensor(
        tf2_build_neighbor_list(
            coord,
            atype,
            nloc,
            rcut,
            sel,
            distinguish_types=distinguish_types,
        )
    )


def extend_coord_with_ghosts(
    coord: Any,
    atype: Any,
    cell: Any | None,
    rcut: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return tuple(
        to_tf_tensor(value)
        for value in tf2_extend_coord_with_ghosts(coord, atype, cell, rcut)
    )
