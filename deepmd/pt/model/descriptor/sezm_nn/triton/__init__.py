# SPDX-License-Identifier: LGPL-3.0-or-later
"""Public Triton entry points for SeZM SO(2) rotations."""

from .autograd import (
    edge_geometry_rbf_triton,
    rotate_back_triton,
    rotate_to_local_triton,
)
from .constants import (
    SEZM_TRITON_AVAILABLE,
    TritonRotationMode,
)
from .dispatch import (
    resolve_triton_rotation_mode,
    sezm_triton_enabled,
    uses_triton_kernel,
)

__all__ = [
    "SEZM_TRITON_AVAILABLE",
    "TritonRotationMode",
    "edge_geometry_rbf_triton",
    "resolve_triton_rotation_mode",
    "rotate_back_triton",
    "rotate_to_local_triton",
    "sezm_triton_enabled",
    "uses_triton_kernel",
]
