# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hardware-accelerated SeZM/DPA4 operators.

This package hosts ``make_fx``-composable Triton implementations of SeZM hot
paths.  Kernel entry points are internal implementation details of the SeZM
descriptor; the package-level API only exposes availability.
"""

from .radial_mix import (
    RADIAL_MIX_TRITON_AVAILABLE,
)
from .so2_rotation import (
    TRITON_ROTATION_AVAILABLE,
)

# Both kernel modules guard their ``@triton.jit`` definitions behind a ``triton``
# import, so the two module-level checks are equivalent. Expose a single
# package-level availability flag.
TRITON_AVAILABLE = TRITON_ROTATION_AVAILABLE and RADIAL_MIX_TRITON_AVAILABLE

__all__ = [
    "TRITON_AVAILABLE",
]
