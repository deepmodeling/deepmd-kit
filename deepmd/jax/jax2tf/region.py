# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility wrappers for TensorFlow region helpers."""

from typing import (
    Any,
)

import tensorflow as tf

from deepmd.tf2.common import (
    to_tf_tensor,
)
from deepmd.tf2.utils._dpmodel import inter2phys as tf2_inter2phys
from deepmd.tf2.utils._dpmodel import normalize_coord as tf2_normalize_coord
from deepmd.tf2.utils._dpmodel import to_face_distance as tf2_to_face_distance

__all__ = [
    "inter2phys",
    "normalize_coord",
    "to_face_distance",
]


def inter2phys(coord: Any, cell: Any) -> tf.Tensor:
    return to_tf_tensor(tf2_inter2phys(coord, cell))


def normalize_coord(coord: Any, cell: Any) -> tf.Tensor:
    return to_tf_tensor(tf2_normalize_coord(coord, cell))


def to_face_distance(cell: Any) -> tf.Tensor:
    return to_tf_tensor(tf2_to_face_distance(cell))
