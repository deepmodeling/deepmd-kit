# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for the array API."""

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


def xp_asarray_nodetach(
    xp: Any,
    obj: Any,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Array:
    """``xp.asarray`` that preserves autograd for backend tensors.

    ``torch.asarray`` detaches its input from the autograd graph, so calling
    ``xp.asarray`` on a weight attribute that is already a backend tensor
    (e.g. a ``torch.nn.Parameter`` registered by the pt_expt backend)
    silently breaks gradient flow to that weight.  This helper converts
    genuine non-backend data (numpy arrays, python scalars/lists) via
    ``xp.asarray``; backend tensors are returned as-is, with an optional
    differentiable dtype cast via ``xp.astype``.

    The ``device`` argument only applies to the conversion path: backend
    tensors are assumed to already live on the working device (they are
    created together with the inputs).
    """
    if isinstance(obj, np.ndarray) or not array_api_compat.is_array_api_obj(obj):
        if dtype is None:
            return xp.asarray(obj, device=device)
        return xp.asarray(obj, dtype=dtype, device=device)
    if dtype is not None and obj.dtype != dtype:
        obj = xp.astype(obj, dtype)
    return obj


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
    # torch.take_along_dim requires int64 indices
    if array_api_compat.is_torch_array(indices):
        indices = xp.astype(indices, xp.int64)
    if array_api_compat.is_torch_array(arr):
        # Use torch.gather directly for torch.export dynamic shape compatibility.
        # array_api_compat's take_along_axis / torch.take_along_dim specializes
        # the source dimension size to a constant during torch.export tracing,
        # breaking dynamic shape export.  torch.gather is the underlying
        # primitive and handles symbolic shapes correctly.
        import torch

        return torch.gather(arr, axis, indices)
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

    dev = array_api_compat.device(indices)
    offset = (xp.arange(indices.shape[0], dtype=indices.dtype, device=dev) * m)[
        :, xp.newaxis
    ]
    indices = xp.reshape(offset + indices, (-1,))

    out = xp.take(arr, indices)
    out = xp.reshape(out, shape)
    return xp_swapaxes(out, axis, -1)


def xp_take_first_n(arr: Array, dim: int, n: int) -> Array:
    """Take the first *n* elements along *dim*.

    For torch tensors, uses ``torch.index_select`` so that
    ``torch.export`` does not emit a contiguity guard that would
    prevent the ``nall == nloc`` (no-PBC) case from working.
    For numpy / jax, uses regular slicing.
    """
    if array_api_compat.is_torch_array(arr):
        import torch

        indices = torch.arange(n, dtype=torch.int64, device=arr.device)
        return torch.index_select(arr, dim, indices)
    slices = [slice(None)] * arr.ndim
    slices[dim] = slice(0, n)
    return arr[tuple(slices)]


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
    if getattr(xp, "__name__", "") == "deepmd._vendors.ndtensorflow":
        import tensorflow as tf

        input_tensor = input.unwrap()
        index_tensor = tf.cast(index.unwrap(), tf.int64)
        src_tensor = src.unwrap()
        rank = input_tensor.shape.rank
        if rank is None:
            raise ValueError("xp_scatter_sum requires a statically known rank")
        dim = dim + rank if dim < 0 else dim
        src_shape = tf.shape(src_tensor, out_type=tf.int64)
        coords = []
        for axis in range(rank):
            if axis == dim:
                coord = index_tensor
            else:
                view_shape = [1] * rank
                view_shape[axis] = src_shape[axis]
                coord = tf.broadcast_to(
                    tf.reshape(tf.range(src_shape[axis], dtype=tf.int64), view_shape),
                    src_shape,
                )
            coords.append(coord)
        scatter_indices = tf.reshape(tf.stack(coords, axis=-1), (-1, rank))
        scatter_updates = tf.reshape(src_tensor, (-1,))
        scattered = tf.scatter_nd(
            scatter_indices,
            scatter_updates,
            tf.shape(input_tensor, out_type=tf.int64),
        )
        return xp.asarray(input_tensor + scattered)

    # Create flat index array matching input shape
    idx = xp.arange(input.size, dtype=xp.int64, device=array_api_compat.device(input))
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
    elif getattr(xp, "__name__", "") == "deepmd._vendors.ndtensorflow":
        import tensorflow as tf

        x_tensor = x.unwrap()
        indices_tensor = tf.reshape(tf.cast(indices.unwrap(), tf.int64), (-1, 1))
        values_tensor = values.unwrap()
        updates = tf.scatter_nd(
            indices_tensor,
            values_tensor,
            tf.shape(x_tensor, out_type=tf.int64),
        )
        return xp.asarray(x_tensor + updates)
    else:
        # Fallback for array_api_strict: use basic indexing only
        # may need a more efficient way to do this
        n = indices.shape[0]
        for i in range(n):
            idx = int(indices[i])
            x[idx, ...] = x[idx, ...] + values[i, ...]
        return x


def xp_hint_dynamic_size(x: Array) -> None:
    """Mark a data-dependent leading dimension as a valid size for torch.export.

    Under symbolic tracing (``make_fx`` / ``torch.export``) the length of a
    data-dependent array (e.g. the output of ``nonzero`` or a tensor-``repeat``)
    is an UNBACKED SymInt; guarding Python control flow or allocations on it
    raises ``GuardOnDataDependentSymNode``. ``torch._check_is_size`` registers
    the ``>= 0`` size hint that lets the tracer treat it as a proper dimension
    (recorded as a ``sym_constrain_range_for_size`` node, preserved by AOTI).

    No-op for numpy / jax / eager-torch concrete shapes — safe to call
    unconditionally from dpmodel code (torch imported lazily, torch arrays only).
    """
    if array_api_compat.is_torch_array(x):
        import torch

        torch._check_is_size(x.shape[0])


def xp_maximum_at(x: Array, indices: Array, values: Array) -> Array:
    """Segment max-assign of values into x at the specified indices.

    Element-wise analogue of :func:`xp_add_at` that takes the maximum instead
    of the sum: for every ``k`` it assigns ``x[indices[k]] = maximum(
    x[indices[k]], values[k])``. Repeated indices reduce to the per-segment
    maximum, which is order-independent.

    Parameters
    ----------
    x : Array
        Destination array indexed along axis 0; typically pre-filled with
        ``-inf`` so empty segments stay neutral.
    indices : Array
        Integer destination indices with shape (K,).
    values : Array
        Source values with shape (K, *x.shape[1:]).

    Returns
    -------
    Array
        The updated array (modified in place and returned for NumPy; a new
        array for JAX/PyTorch).
    """
    xp = array_api_compat.array_namespace(x, indices, values)
    if array_api_compat.is_numpy_array(x):
        # NumPy: in-place ufunc reduction at the given indices.
        xp.maximum.at(x, indices, values)
        return x

    elif array_api_compat.is_jax_array(x):
        # JAX: functional indexed-max update, not in-place.
        return x.at[indices].max(values)
    elif array_api_compat.is_torch_array(x):
        import torch

        index = indices.reshape([-1] + [1] * (values.ndim - 1)).expand_as(values)
        return torch.scatter_reduce(
            x, 0, index, values, reduce="amax", include_self=True
        )
    elif getattr(xp, "__name__", "") == "deepmd._vendors.ndtensorflow":
        import tensorflow as tf

        x_tensor = x.unwrap()
        indices_tensor = tf.reshape(tf.cast(indices.unwrap(), tf.int64), (-1,))
        values_tensor = values.unwrap()
        reduced = tf.math.unsorted_segment_max(
            values_tensor,
            indices_tensor,
            tf.shape(x_tensor, out_type=tf.int64)[0],
        )
        return xp.asarray(tf.maximum(x_tensor, reduced))
    else:
        # Fallback for array_api_strict: basic indexing only.
        n = indices.shape[0]
        for i in range(n):
            idx = int(indices[i])
            x[idx, ...] = xp.maximum(x[idx, ...], values[i, ...])
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

    For JAX and PyTorch arrays, returns a new array (non-mutating).
    For NumPy arrays, modifies in-place and returns the same array.

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
        Modified array (new array for JAX/PyTorch, same array for NumPy)
    """
    if array_api_compat.is_jax_array(x):
        # JAX doesn't support in-place item assignment
        return x.at[mask].set(values)
    elif array_api_compat.is_torch_array(x):
        # PyTorch: clone to avoid mutating the input (non-mutating version)
        import torch

        result = torch.clone(x)
        result[mask] = values
        return result
    # Standard item assignment for NumPy, array-api-strict, etc.
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
        result = xp.zeros(
            (max(minlength, int(xp.max(x)) + 1),),
            dtype=weights.dtype,
            device=array_api_compat.device(weights),
        )
        result = xp_add_at(result, x, weights)
    return result
