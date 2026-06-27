from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal

import tensorflow as tf

from ._array import Array
from ._namespace import Device, DType, _moveaxis, _unwrap

_Norm = Literal["backward", "ortho", "forward"]


def _wrap(x: tf.Tensor) -> Array:
    return Array._from_tensor(x)


def _complex_dtype(dtype: DType) -> DType:
    if dtype in (tf.float64, tf.complex128):
        return tf.complex128
    return tf.complex64


def _real_dtype(dtype: DType) -> DType:
    if dtype == tf.complex128:
        return tf.float64
    return tf.float32


def _as_complex(x: tf.Tensor) -> tf.Tensor:
    if x.dtype.is_complex:
        return x
    return tf.cast(x, _complex_dtype(x.dtype))


def _shape_tuple(x: tf.Tensor) -> tuple[int, ...]:
    return tuple(x.shape.as_list())


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def _normalize_axes(
    axes: Sequence[int] | None,
    ndim: int,
    s: Sequence[int] | None,
) -> tuple[int, ...]:
    if axes is None:
        axes = tuple(range(ndim)) if s is None else tuple(range(ndim - len(s), ndim))
    axes = tuple(_normalize_axis(a, ndim) for a in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    return axes


def _resize_axis(x: tf.Tensor, n: int | None, axis: int) -> tf.Tensor:
    if n is None or x.shape[axis] == n:
        return x
    shape = list(_shape_tuple(x))
    current = shape[axis]
    if current > n:
        begin = [0] * x.shape.rank
        size = shape
        size[axis] = n
        return tf.slice(x, begin, size)
    paddings = [[0, 0] for _ in range(x.shape.rank)]
    paddings[axis][1] = n - current
    return tf.pad(x, paddings)


def _apply_1d(x: tf.Tensor, func, n: int | None, axis: int) -> tf.Tensor:
    x = _moveaxis(x, axis, -1)
    x = _resize_axis(x, n, -1)
    x = func(x)
    return _moveaxis(x, -1, axis)


def _norm_size(x: tf.Tensor, axes: tuple[int, ...], s: Sequence[int] | None) -> int:
    if s is None:
        return math.prod(x.shape[a] for a in axes)
    return math.prod(s)


def _scale_forward(x: tf.Tensor, n: int, norm: _Norm) -> tf.Tensor:
    if norm == "backward":
        return x
    scale = tf.cast(n if norm == "forward" else math.sqrt(n), x.dtype)
    return x / scale


def _scale_inverse(x: tf.Tensor, n: int, norm: _Norm) -> tf.Tensor:
    if norm == "backward":
        return x
    scale = tf.cast(n if norm == "forward" else math.sqrt(n), x.dtype)
    return x * scale


def fft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    tensor = _as_complex(_unwrap(x))
    axis = _normalize_axis(axis, tensor.shape.rank)
    out = _apply_1d(tensor, tf.signal.fft, n, axis)
    return _wrap(_scale_forward(out, out.shape[axis], norm))


def ifft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    tensor = _as_complex(_unwrap(x))
    axis = _normalize_axis(axis, tensor.shape.rank)
    out = _apply_1d(tensor, tf.signal.ifft, n, axis)
    return _wrap(_scale_inverse(out, out.shape[axis], norm))


def fftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    tensor = _as_complex(_unwrap(x))
    axes_ = _normalize_axes(axes, tensor.shape.rank, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    n = _norm_size(tensor, axes_, s)
    out = tensor
    for axis, size in zip(axes_, sizes, strict=True):
        out = _apply_1d(out, tf.signal.fft, size, axis)
    return _wrap(_scale_forward(out, n, norm))


def ifftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    tensor = _as_complex(_unwrap(x))
    axes_ = _normalize_axes(axes, tensor.shape.rank, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    n = _norm_size(tensor, axes_, s)
    out = tensor
    for axis, size in zip(axes_, sizes, strict=True):
        out = _apply_1d(out, tf.signal.ifft, size, axis)
    return _wrap(_scale_inverse(out, n, norm))


def rfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    tensor = _unwrap(x)
    axis = _normalize_axis(axis, tensor.shape.rank)
    out = _apply_1d(tensor, tf.signal.rfft, n, axis)
    size = n if n is not None else tensor.shape[axis]
    return _wrap(_scale_forward(out, size, norm))


def irfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    tensor = _unwrap(x)
    axis = _normalize_axis(axis, tensor.shape.rank)
    out = _apply_1d(
        tensor,
        lambda y: tf.signal.irfft(y, fft_length=[n] if n is not None else None),
        None,
        axis,
    )
    size = n if n is not None else out.shape[axis]
    return _wrap(_scale_inverse(out, size, norm))


def rfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    tensor = _unwrap(x)
    axes_ = _normalize_axes(axes, tensor.shape.rank, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    n = _norm_size(tensor, axes_, s)
    out = _apply_1d(tensor, tf.signal.rfft, sizes[-1], axes_[-1])
    for axis, size in zip(axes_[:-1], sizes[:-1], strict=True):
        out = _apply_1d(_as_complex(out), tf.signal.fft, size, axis)
    return _wrap(_scale_forward(out, n, norm))


def irfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    tensor = _unwrap(x)
    axes_ = _normalize_axes(axes, tensor.shape.rank, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    if s is None:
        last_size = 2 * (tensor.shape[axes_[-1]] - 1)
        n = math.prod([*(tensor.shape[a] for a in axes_[:-1]), last_size])
    else:
        n = math.prod(s)
    out = tensor
    for axis, size in zip(axes_[:-1], sizes[:-1], strict=True):
        out = _apply_1d(out, tf.signal.ifft, size, axis)
    out = _apply_1d(
        out,
        lambda y: tf.signal.irfft(
            y, fft_length=[sizes[-1]] if sizes[-1] is not None else None
        ),
        None,
        axes_[-1],
    )
    return _wrap(_scale_inverse(out, n, norm))


def hfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    tensor = _unwrap(x)
    size = n if n is not None else 2 * (tensor.shape[axis] - 1)
    return _wrap(
        _unwrap(irfft(_wrap(tf.math.conj(tensor)), n=size, axis=axis, norm=norm))
        * tf.cast(size, _real_dtype(tensor.dtype))
    )


def ihfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    tensor = _unwrap(x)
    size = n if n is not None else tensor.shape[axis]
    return _wrap(
        tf.math.conj(_unwrap(rfft(x, n=size, axis=axis, norm=norm)))
        / tf.cast(size, _complex_dtype(tensor.dtype))
    )


def fftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    with tf.device(device) if device is not None else nullcontext():
        dtype = dtype or tf.float32
        positive = tf.range(0, (n - 1) // 2 + 1, dtype=dtype)
        negative = tf.range(-(n // 2), 0, dtype=dtype)
        return _wrap(tf.concat([positive, negative], axis=0) / tf.cast(n * d, dtype))


def rfftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    with tf.device(device) if device is not None else nullcontext():
        dtype = dtype or tf.float32
        return _wrap(tf.range(0, n // 2 + 1, dtype=dtype) / tf.cast(n * d, dtype))


def fftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> Array:
    tensor = _unwrap(x)
    axes_ = _normalize_axes(
        None if axes is None else (axes if isinstance(axes, Sequence) else (axes,)),
        tensor.shape.rank,
        None,
    )
    shifts = tuple(tensor.shape[axis] // 2 for axis in axes_)
    return _wrap(tf.roll(tensor, shifts, axes_))


def ifftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> Array:
    tensor = _unwrap(x)
    axes_ = _normalize_axes(
        None if axes is None else (axes if isinstance(axes, Sequence) else (axes,)),
        tensor.shape.rank,
        None,
    )
    shifts = tuple(-(tensor.shape[axis] // 2) for axis in axes_)
    return _wrap(tf.roll(tensor, shifts, axes_))


__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]


def __dir__() -> list[str]:
    return __all__
