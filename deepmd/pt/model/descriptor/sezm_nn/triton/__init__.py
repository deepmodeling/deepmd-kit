# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hardware-accelerated SeZM/DPA4 operators.

This package hosts ``make_fx``-composable Triton implementations of SeZM hot
paths.  Kernel entry points are internal implementation details of the SeZM
descriptor; the package-level API only exposes availability.
"""

from .so2_rotation import (
    TRITON_ROTATION_AVAILABLE,
)

__all__ = [
    "TRITON_ROTATION_AVAILABLE",
]
