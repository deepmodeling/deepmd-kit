# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hardware-accelerated SeZM/DPA4 operators.

This package hosts ``make_fx``-composable Triton implementations of SeZM hot
paths.  The SO(2) rotation API exposes a general dense path that honors arbitrary
coefficient indices and a block path for the canonical m-major ``mmax=1`` layout.
"""

from .so2_rotation import (
    rotate_back_block,
    rotate_back_dense,
    rotate_to_local_block,
    rotate_to_local_dense,
)

__all__ = [
    "rotate_back_block",
    "rotate_back_dense",
    "rotate_to_local_block",
    "rotate_to_local_dense",
]
