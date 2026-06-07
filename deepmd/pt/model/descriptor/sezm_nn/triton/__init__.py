# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hardware-accelerated SeZM/DPA4 operators.

This package hosts clean, ``torch.compile``-composable Triton implementations of
SeZM hot paths. The first member is the fused SO(2)/Wigner rotation pair used by
the SO(2) convolution (``rotate_to_local`` / ``rotate_back``).
"""

from .so2_rotation import (
    rotate_back,
    rotate_to_local,
)

__all__ = [
    "rotate_back",
    "rotate_to_local",
]
