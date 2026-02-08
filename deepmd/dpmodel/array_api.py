# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for the array API."""

from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np
from packaging.version import (
    Version,
)

# Type alias for array_api compatible arrays
Array = np.ndarray | Any  # Any to support JAX, PyTorch, etc. arrays


def support_array_api(version: str) -> Callable:
    """Mark a function as supporting the specific version of the array API.

    Parameters
    ----------
    version : str
        The version of the array API

    Returns
    -------
    Callable
        The decorated function

    Examples
    --------
    >>> @support_array_api(version="2022.12")
    ... def f(x):
    ...     pass
    """

    def set_version(func: Callable) -> Callable:
        func.array_api_version = version
        return func

    return set_version


# array api adds take_along_axis in https://github.com/data-apis/array-api/pull/816
# but it hasn't been released yet
# below is a pure Python implementation of take_along_axis
# https://github.com/data-apis/array-api/issues/177#issuecomment-2093630595
def xp_swapaxes(a: Array, axis1: int, axis2: int) -> Array:
    xp = array_api_compat.array_namespace(a)
    axes = list(range(a.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    a = xp.permute_dims(a, axes)
    return a


def xp_take_along_axis(arr: Array, indices: Array, axis: int) -> Array:
    xp = array_api_compat.array_namespace(arr)
    if Version(xp.__array_api_version__) >= Version("2024.12"):
        # see: https://github.com/data-apis/array-api-strict/blob/d086c619a58f35c38240592ef994aa19ca7beebc/array_api_strict/_indexing_functions.py#L30-L39
        return xp.take_along_axis(arr, indices, axis=axis)
    arr = xp_swapaxes(arr, axis, -1)
    indices = xp_swapaxes(indices, axis, -1)

    m = arr.shape[-1]
    n = indices.shape[-1]

    shape = list(arr.shape)
    shape.pop(-1)
    shape = (*shape, n)

    arr = xp.reshape(arr, (-1,))
    if n != 0:
        indices = xp.reshape(indices, (-1, n))
    else:
        indices = xp.reshape(indices, (0, 0))

    offset = (xp.arange(indices.shape[0], dtype=indices.dtype) * m)[:, xp.newaxis]
    indices = xp.reshape(offset + indices, (-1,))

    out = xp.take(arr, indices)
    out = xp.reshape(out, shape)
    return xp_swapaxes(out, axis, -1)


def xp_scatter_sum(input: Array, dim: int, index: Array, src: Array) -> Array:
    """Reduces all values from the src tensor to the indices specified in the index tensor.

    This function is similar to PyTorch's scatter_add and JAX's scatter_sum.
    It adds values from src to input at positions specified by index along the given dimension.
    """
    if array_api_compat.is_torch_array(input):
        # PyTorch: use scatter_add (non-mutating version) for better performance
        import torch

        return torch.scatter_add(input, dim, index, src)

    # Generic array_api implementation (works for JAX, NumPy, array-api-strict, etc.)
    xp = array_api_compat.array_namespace(input)

    # Create flat index array matching input shape
    idx = xp.arange(input.size, dtype=xp.int64)
    idx = xp.reshape(idx, input.shape)

    # Get flat indices where we want to add values
    new_idx = xp_take_along_axis(idx, index, axis=dim)
    new_idx = xp.reshape(new_idx, (-1,))

    # Flatten arrays
    shape = input.shape
    input_flat = xp.reshape(input, (-1,))
    src_flat = xp.reshape(src, (-1,))

    # Add values at the specified indices
    result = xp_add_at(input_flat, new_idx, src_flat)

    # Reshape back to original shape
    return xp.reshape(result, shape)


def xp_add_at(x: Array, indices: Array, values: Array) -> Array:
    """Adds values to the specified indices of x in place or returns new x (for JAX)."""
    xp = array_api_compat.array_namespace(x, indices, values)
    if array_api_compat.is_numpy_array(x):
        # NumPy: supports np.add.at (in-place)
        xp.add.at(x, indices, values)
        return x

    elif array_api_compat.is_jax_array(x):
        # JAX: functional update, not in-place
        return x.at[indices].add(values)
    elif array_api_compat.is_torch_array(x):
        # PyTorch: use index_add (non-mutating version)
        import torch

        return torch.index_add(x, 0, indices, values)
    else:
        # Fallback for array_api_strict: use basic indexing only
        # may need a more efficient way to do this
        n = indices.shape[0]
        for i in range(n):
            idx = int(indices[i])
            x[idx, ...] = x[idx, ...] + values[i, ...]
        return x


def xp_sigmoid(x: Array) -> Array:
    """Compute the sigmoid function.

    JAX and PyTorch have optimized sigmoid implementations.
    See https://github.com/jax-ml/jax/discussions/15617
    """
    if array_api_compat.is_jax_array(x):
        from deepmd.jax.env import (
            jax,
        )

        return jax.nn.sigmoid(x)
    elif array_api_compat.is_torch_array(x):
        import torch

        return torch.sigmoid(x)
    xp = array_api_compat.array_namespace(x)
    return 1 / (1 + xp.exp(-x))


def xp_setitem_at(x: Array, mask: Array, values: Array) -> Array:
    """Set items at boolean mask indices.

    For JAX arrays, uses functional .at[mask].set() syntax.
    For other arrays, uses standard item assignment.

    Parameters
    ----------
    x : Array
        The array to modify
    mask : Array
        Boolean mask indicating positions to set
    values : Array
        Values to set at masked positions

    Returns
    -------
    Array
        Modified array (new array for JAX, same array for others)
    """
    if array_api_compat.is_jax_array(x):
        # JAX doesn't support in-place item assignment
        return x.at[mask].set(values)
    # Standard item assignment for NumPy, PyTorch, etc.
    x[mask] = values
    return x


def xp_bincount(x: Array, weights: Array | None = None, minlength: int = 0) -> Array:
    """Counts the number of occurrences of each value in x."""
    xp = array_api_compat.array_namespace(x)
    if (
        array_api_compat.is_numpy_array(x)
        or array_api_compat.is_jax_array(x)
        or array_api_compat.is_torch_array(x)
    ):
        result = xp.bincount(x, weights=weights, minlength=minlength)
    else:
        if weights is None:
            weights = xp.ones_like(x)
        result = xp.zeros((max(minlength, int(xp.max(x)) + 1),), dtype=weights.dtype)
        result = xp_add_at(result, x, weights)
    return result
