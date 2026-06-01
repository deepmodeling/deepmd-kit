# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared constants and feature flags for SeZM Triton kernels."""

from __future__ import (
    annotations,
)

from enum import (
    IntEnum,
)

import torch

_HAS_TORCH_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)

if _HAS_TORCH_TRITON_OP:
    try:
        import triton  # noqa: F401
    except ImportError:
        SEZM_TRITON_AVAILABLE = False
    else:
        SEZM_TRITON_AVAILABLE = True
else:
    SEZM_TRITON_AVAILABLE = False

# Triton dot kernels require K >= 16 on the current CUDA backend.
TRITON_GRID_E_STRIDE = 2048
TRITON_BLOCK_FULL = 16
TRITON_BLOCK_REDUCED = 16
TRITON_BLOCK_CHANNEL = 32
TRITON_SMALL_BLOCK_CHANNEL = 128
TRITON_SMALL_FULL_DIM = 16
TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE = 128
TRITON_EDGE_GEOMETRY_RBF_BLOCK_RADIAL = 16


class TritonRotationMode(IntEnum):
    """Dispatch mode for the SeZM rotation hot path."""

    GENERIC_TILED = 0
    SMALL_LE1 = 1
    SMALL_L2 = 2
    SMALL_L3 = 3
    EAGER_REFERENCE = 4
