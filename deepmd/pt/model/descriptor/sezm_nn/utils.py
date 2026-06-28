# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Utility helpers for the SeZM descriptor package.

This module provides small numerical helpers, dtype conversion utilities, and
profiling helpers shared across the SeZM descriptor implementation.
"""

from __future__ import (
    annotations,
)

import math
import os
from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch
import torch.nn as nn

from deepmd.pt.utils.utils import (
    get_generator,
)

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
    )

ATTN_RES_MODES = ("none", "independent", "dependent")

_INFER_TRUE = ("1", "true", "yes", "on")


def use_triton_infer() -> bool:
    """Return whether the opt-in Triton inference kernels are enabled.

    The flag is controlled by the ``DP_TRITON_INFER`` environment variable and
    is read at module construction time so that it becomes a compile-time
    constant in the traced (``make_fx``) graph. It only takes effect during
    inference; training always uses the dense reference path.

    Returns
    -------
    bool
        ``True`` when ``DP_TRITON_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_TRITON_INFER", "0").strip().lower() in _INFER_TRUE


def use_amp_infer() -> bool:
    """Return whether bf16 autocast is enabled for inference.

    The flag is controlled by the ``DP_AMP_INFER`` environment variable and is
    read at module construction time. It only affects inference when the
    descriptor's ``use_amp`` option is also enabled; training follows
    ``use_amp`` regardless of this environment variable.

    Returns
    -------
    bool
        ``True`` when ``DP_AMP_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_AMP_INFER", "0").strip().lower() in _INFER_TRUE


def init_trunc_normal_fan_in_out(
    weight: torch.Tensor,
    seed: int | list[int] | None,
    scale: float = 1.0,
) -> None:
    """Initialize weight with truncated normal distribution.

    Uses Xavier-like variance scaling: std = scale / sqrt(fan_in + fan_out).
    Truncation at +/-3*std prevents extreme outliers.

    Parameters
    ----------
    weight : torch.Tensor
        Weight tensor with shape (out_features, in_features).
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
    nn.init.trunc_normal_(
        weight,
        mean=0.0,
        std=std,
        a=-3.0 * std,
        b=3.0 * std,
        generator=get_generator(seed),
    )


@contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    """
    Create an NVTX range when CUDA is available; otherwise, no-op.

    Parameters
    ----------
    name
        Range name shown in Nsight Systems/Compute.
    """
    if torch.cuda.is_available():
        nvtx = torch.cuda.nvtx
        if hasattr(nvtx, "range"):
            with nvtx.range(name):
                yield
            return
    yield


def safe_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute vector norm with smooth epsilon regularization.

    Uses float32 for computation when input is fp16/bf16.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (N, 3), where N is the number of vectors.
    eps : float
        Lower bound for the norm.

    Returns
    -------
    torch.Tensor
        Norm with shape (N, 1).
    """
    in_dtype = x.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    eps_sq = float(eps) * float(eps)
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps_sq)
    return norm.to(dtype=in_dtype)


def safe_numpy_to_tensor(
    data: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, np.ndarray):
        # Handle bfloat16: numpy uses ml_dtypes.bfloat16, which torch.as_tensor
        # cannot convert. Convert to float32 first, then cast to target dtype.
        if hasattr(data.dtype, "name") and "bfloat16" in data.dtype.name:
            data = data.astype(np.float32)
        return torch.as_tensor(data, device=device).to(dtype)
    return torch.as_tensor(data, device=device, dtype=dtype)


def get_promoted_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Get promoted dtype for numerical stability.

    For bf16/fp16, use float32 to ensure numerical stability
    in computation and storage compatibility.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def np_safe(
    tensor: torch.Tensor | None,
) -> np.ndarray | None:
    """
    Convert tensor to numpy array, promoting low-precision types to fp32.

    For bf16/fp16, converts to fp32 first since NumPy/HDF5 do not natively
    support these formats. fp32/fp64 are kept unchanged.

    Parameters
    ----------
    tensor
        PyTorch tensor to convert. Can be None.

    Returns
    -------
    np.ndarray or None
        numpy array with at least fp32 precision.
    """
    if tensor is None:
        return None
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()
