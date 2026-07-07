# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Utility helpers for the DPA4/SeZM descriptor package.

This module provides small numerical helpers and dtype conversion utilities
shared across the SeZM descriptor implementation.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.utils``.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import array_api_compat
import numpy as np

ATTN_RES_MODES = ("none", "independent", "dependent")


def init_trunc_normal_fan_in_out(
    weight: np.ndarray,
    seed: int | list[int] | None,
    scale: float = 1.0,
) -> None:
    """Initialize weight with truncated normal distribution.

    Uses Xavier-like variance scaling: std = scale / sqrt(fan_in + fan_out).
    Truncation at +/-3*std prevents extreme outliers.

    Parameters
    ----------
    weight : np.ndarray
        Weight array with shape (out_features, in_features).
    seed : int | list[int] | None
        Random seed for reproducibility.
    scale : float, default=1.0
        Multiplicative scale factor in the standard deviation numerator.
    """
    if weight.ndim != 2:
        raise ValueError("`weight` must be a 2D tensor")
    if scale <= 0:
        raise ValueError("`scale` must be positive")
    fan_out, fan_in = weight.shape
    std = float(scale) / math.sqrt(fan_in + fan_out)
    rng = np.random.default_rng(seed)
    # Rejection sampling reproduces the truncated normal on [-3*std, 3*std].
    values = rng.normal(0.0, std, size=weight.shape)
    out_of_bounds = np.abs(values) > 3.0 * std
    while out_of_bounds.any():
        values[out_of_bounds] = rng.normal(
            0.0, std, size=int(np.count_nonzero(out_of_bounds))
        )
        out_of_bounds = np.abs(values) > 3.0 * std
    weight[...] = values.astype(weight.dtype, copy=False)


def safe_norm(x: Any, eps: float = 1e-7) -> Any:
    """
    Compute vector norm with smooth epsilon regularization.

    Uses float32 for computation when input is fp16/bf16.

    Parameters
    ----------
    x : Array
        Input array with shape (N, 3), where N is the number of vectors.
    eps : float
        Lower bound for the norm.

    Returns
    -------
    Array
        Norm with shape (N, 1).
    """
    xp = array_api_compat.array_namespace(x)
    in_dtype = x.dtype
    # ``str(dtype)`` matches both "float16" and "bfloat16" across namespaces.
    promote = "float16" in str(in_dtype)
    if promote:
        x = xp.astype(x, xp.float32)
    eps_sq = float(eps) * float(eps)
    norm = xp.sqrt(xp.sum(x * x, axis=-1, keepdims=True) + eps_sq)
    if promote:
        norm = xp.astype(norm, in_dtype)
    return norm


def get_promoted_dtype(dtype: Any) -> Any:
    """
    Get promoted dtype for numerical stability.

    For bf16/fp16, use float32 to ensure numerical stability
    in computation and storage compatibility.
    """
    name = getattr(dtype, "name", None) or str(dtype)
    if "float16" in name:  # matches float16 and bfloat16
        return np.dtype(np.float32)
    return dtype
