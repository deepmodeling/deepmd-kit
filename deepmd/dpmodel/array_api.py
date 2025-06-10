# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for the array API."""

import array_api_compat
import numpy as np
from packaging.version import (
    Version,
)


def support_array_api(version: str) -> callable:
    """Mark a function as supporting the specific version of the array API.

    Parameters
    ----------
    version : str
        The version of the array API

    Returns
    -------
    callable
        The decorated function

    Examples
    --------
    >>> @support_array_api(version="2022.12")
    ... def f(x):
    ...     pass
    """

    def set_version(func: callable) -> callable:
        func.array_api_version = version
        return func

    return set_version


# array api adds take_along_axis in https://github.com/data-apis/array-api/pull/816
# but it hasn't been released yet
# below is a pure Python implementation of take_along_axis
# https://github.com/data-apis/array-api/issues/177#issuecomment-2093630595
def xp_swapaxes(a, axis1, axis2):
    xp = array_api_compat.array_namespace(a)
    axes = list(range(a.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    a = xp.permute_dims(a, axes)
    return a


def xp_take_along_axis(arr, indices, axis):
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
    shape = [*shape, n]

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


def xp_scatter_sum(input, dim, index: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Reduces all values from the src tensor to the indices specified in the index tensor."""
    # jax only
    if array_api_compat.is_jax_array(input):
        from deepmd.jax.common import (
            scatter_sum,
        )

        return scatter_sum(
            input,
            dim,
            index,
            src,
        )
    else:
        raise NotImplementedError("Only JAX arrays are supported.")


def xp_add_at(x, indices, values):
    """Adds values to the specified indices of x in place or returns new x (for JAX)."""
    xp = array_api_compat.array_namespace(x, indices, values)
    if array_api_compat.is_numpy_array(x):
        # NumPy: supports np.add.at (in-place)
        xp.add.at(x, indices, values)
        return x

    elif array_api_compat.is_jax_array(x):
        # JAX: functional update, not in-place
        return x.at[indices].add(values)
    else:
        # Fallback for array_api_strict: use basic indexing only
        # may need a more efficient way to do this
        n = indices.shape[0]
        for i in range(n):
            idx = int(indices[i])
            x[idx, ...] = x[idx, ...] + values[i, ...]
        return x


def xp_bincount(x, weights=None, minlength=0):
    """Counts the number of occurrences of each value in x."""
    xp = array_api_compat.array_namespace(x)
    if array_api_compat.is_numpy_array(x) or array_api_compat.is_jax_array(x):
        result = xp.bincount(x, weights=weights, minlength=minlength)
    else:
        if weights is None:
            weights = xp.ones_like(x)
        result = xp.zeros((max(minlength, int(xp.max(x)) + 1),), dtype=weights.dtype)
        result = xp_add_at(result, x, weights)
    return result
