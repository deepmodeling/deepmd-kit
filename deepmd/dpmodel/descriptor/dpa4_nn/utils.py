# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Utility helpers for the DPA4/SeZM descriptor package.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.utils``.
It provides the small numeric helpers shared across the DPA4 descriptor
implementation.

Init-time helpers (``init_trunc_normal_fan_in_out``) operate on static numpy
data and are plain numpy by design (not array-API). ``safe_norm`` operates on
runtime tensors and is array-API compatible.

Helpers from the pt version intentionally NOT ported:

- ``nvtx_range``: CUDA profiling, torch-only.
- ``use_triton_infer``: Triton inference kernels, torch-only.
- ``safe_numpy_to_tensor``: numpy -> torch conversion glue; dpmodel code uses
  ``xp.asarray`` directly.
- ``np_safe``: torch -> numpy conversion glue; dpmodel code uses
  ``deepmd.dpmodel.common.to_numpy_array`` instead.

``get_promoted_dtype`` IS ported (numpy equivalent) because core modules use it
to pick a stable computation/storage dtype.
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

    NumPy equivalent of the pt version: the weight is filled in place from a
    ``np.random.default_rng(seed)`` stream (distribution-equivalent to the
    torch version, not RNG-stream-identical).

    Parameters
    ----------
    weight : np.ndarray
        Weight array with shape (out_features, in_features), modified in place.
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
    # rejection sampling: exact truncated normal on [-3*std, 3*std]
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

    Uses float32 for computation when input is fp16/bf16. This function
    operates on runtime tensors and is array-API compatible.

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
    # matches "float16" and "bfloat16" dtype names across namespaces
    promote = "float16" in str(in_dtype)
    if promote:
        x = xp.astype(x, xp.float32)
    norm = xp.sqrt(xp.sum(x * x, axis=-1, keepdims=True) + float(eps) * float(eps))
    if promote:
        norm = xp.astype(norm, in_dtype)
    return norm


def get_promoted_dtype(dtype: Any) -> Any:
    """
    Get promoted dtype for numerical stability.

    For bf16/fp16, use float32 to ensure numerical stability
    in computation and storage compatibility.

    NumPy equivalent of the pt version; accepts a numpy dtype (including
    ``ml_dtypes.bfloat16``) and returns a numpy dtype.
    """
    name = getattr(dtype, "name", None) or str(dtype)
    if "float16" in name:  # matches float16 and bfloat16
        return np.dtype(np.float32)
    return dtype
