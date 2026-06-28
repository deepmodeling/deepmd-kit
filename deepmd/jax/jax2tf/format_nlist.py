# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility wrappers for TensorFlow neighbor-list formatting."""

from typing import (
    Any,
)

import tensorflow as tf

from deepmd.tf2.common import (
    to_tf_tensor,
)
from deepmd.tf2.utils._dpmodel import format_nlist as tf2_format_nlist

__all__ = ["format_nlist"]


def format_nlist(
    extended_coord: Any,
    nlist: Any,
    nsel: int,
    rcut: float,
) -> tf.Tensor:
    return to_tf_tensor(tf2_format_nlist(extended_coord, nlist, nsel, rcut))
