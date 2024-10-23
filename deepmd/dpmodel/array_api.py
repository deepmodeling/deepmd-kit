# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for the array API."""

import array_api_compat


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
