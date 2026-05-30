# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dispatch helpers for SeZM Triton rotation kernels."""

from __future__ import (
    annotations,
)

from typing import (
    Final,
)

import torch

from .constants import (
    SEZM_TRITON_AVAILABLE,
    TRITON_BLOCK_REDUCED,
    TritonRotationMode,
)

_SMALL_MODE_FROM_DIM: Final[dict[int, TritonRotationMode]] = {
    1: TritonRotationMode.SMALL_LE1,
    4: TritonRotationMode.SMALL_LE1,
    9: TritonRotationMode.SMALL_L2,
    16: TritonRotationMode.SMALL_L3,
}


def coerce_rotation_mode(
    rotation_mode: int | TritonRotationMode,
) -> TritonRotationMode:
    """
    Convert an integer-like dispatch value to ``TritonRotationMode``.

    Parameters
    ----------
    rotation_mode
        Rotation dispatch value.

    Returns
    -------
    TritonRotationMode
        Normalized rotation dispatch mode.
    """
    if isinstance(rotation_mode, TritonRotationMode):
        return rotation_mode
    return TritonRotationMode(int(rotation_mode))


def resolve_triton_rotation_mode(
    *,
    dim_full: int,
    reduced_dim: int,
) -> TritonRotationMode:
    """
    Resolve the SeZM rotation dispatch mode.

    Parameters
    ----------
    dim_full
        Full packed SO(3) dimension.
    reduced_dim
        Truncated m-major coefficient count.

    Returns
    -------
    TritonRotationMode
        Dispatch mode for the current ``(dim_full, reduced_dim)`` pair.

    Raises
    ------
    ValueError
        If either dimension is non-positive.
    """
    dim_full = int(dim_full)
    reduced_dim = int(reduced_dim)
    if dim_full <= 0:
        raise ValueError("dim_full must be positive")
    if reduced_dim <= 0:
        raise ValueError("reduced_dim must be positive")
    base_mode = _SMALL_MODE_FROM_DIM.get(
        dim_full,
        TritonRotationMode.GENERIC_TILED,
    )
    if (
        base_mode == TritonRotationMode.GENERIC_TILED
        and reduced_dim < TRITON_BLOCK_REDUCED
    ):
        return TritonRotationMode.EAGER_REFERENCE
    return base_mode


def sezm_triton_enabled(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    """
    Return whether SeZM should enable the Triton rotation path.

    Parameters
    ----------
    device
        Target device for the rotation path.
    dtype
        Activation dtype for the rotation path.

    Returns
    -------
    bool
        Whether Triton kernels are available for the given device and dtype.
    """
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    return bool(
        SEZM_TRITON_AVAILABLE and device.type == "cuda" and dtype in supported_dtypes
    )


def uses_triton_kernel(
    rotation_mode: int | TritonRotationMode,
) -> bool:
    """
    Return whether the dispatch mode launches a Triton kernel.

    Parameters
    ----------
    rotation_mode
        Rotation dispatch value.

    Returns
    -------
    bool
        ``True`` when the mode launches a Triton kernel instead of eager fallback.
    """
    return coerce_rotation_mode(rotation_mode) != TritonRotationMode.EAGER_REFERENCE
