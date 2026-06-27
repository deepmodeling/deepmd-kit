# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility exports for TensorFlow region helpers."""

from deepmd.tf2.region import (
    b_to_face_distance,
    inter2phys,
    normalize_coord,
    phys2inter,
    to_face_distance,
)

__all__ = [
    "b_to_face_distance",
    "inter2phys",
    "normalize_coord",
    "phys2inter",
    "to_face_distance",
]
