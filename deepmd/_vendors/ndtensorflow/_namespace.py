# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import math
from builtins import abs as py_abs
from builtins import all as py_all
from builtins import any as py_any
from builtins import bool as py_bool
from builtins import max as py_max
from collections import (
    namedtuple,
)
from collections.abc import (
    Sequence,
)
from contextlib import (
    nullcontext,
)
from functools import reduce as _reduce
from functools import wraps as _wraps
from typing import (
    Any,
    Literal,
)

import tensorflow as tf

from ._array import (
    Array,
)

DType = tf.DType
Device = str

bool = tf.bool
int8 = tf.int8
int16 = tf.int16
int32 = tf.int32
int64 = tf.int64
uint8 = tf.uint8
uint16 = tf.uint16
uint32 = tf.uint32
uint64 = tf.uint64
float16 = tf.float16
bfloat16 = tf.bfloat16
float32 = tf.float32
float64 = tf.float64
complex64 = tf.complex64
complex128 = tf.complex128

newaxis = None
e = math.e
inf = math.inf
nan = float("nan")
pi = math.pi

UniqueAllResult = namedtuple(
    "UniqueAllResult", ["values", "indices", "inverse_indices", "counts"]
)
UniqueCountsResult = namedtuple("UniqueCountsResult", ["values", "counts"])
UniqueInverseResult = namedtuple("UniqueInverseResult", ["values", "inverse_indices"])

_py_scalars = (py_bool, int, float, complex)
_bool_dtypes = {tf.bool}
_signed_dtypes = {tf.int8, tf.int16, tf.int32, tf.int64}
_unsigned_dtypes = {tf.uint8, tf.uint16, tf.uint32, tf.uint64}
_real_floating_dtypes = {tf.float16, tf.bfloat16, tf.float32, tf.float64}
_complex_floating_dtypes = {tf.complex64, tf.complex128}
_integral_dtypes = _signed_dtypes | _unsigned_dtypes
_numeric_dtypes = _integral_dtypes | _real_floating_dtypes | _complex_floating_dtypes
_all_dtypes = _bool_dtypes | _numeric_dtypes

_dtype_bits = {
    tf.bool: 1,
    tf.int8: 8,
    tf.int16: 16,
    tf.int32: 32,
    tf.int64: 64,
    tf.uint8: 8,
    tf.uint16: 16,
    tf.uint32: 32,
    tf.uint64: 64,
    tf.float16: 16,
    tf.bfloat16: 16,
    tf.float32: 32,
    tf.float64: 64,
    tf.complex64: 64,
    tf.complex128: 128,
}

_float_for_bits = {
    16: tf.float16,
    32: tf.float32,
    64: tf.float64,
}

_complex_for_bits = {
    32: tf.complex64,
    64: tf.complex128,
}

_finfo = {
    tf.float16: {
        "bits": 16,
        "eps": 0.0009765625,
        "max": 65504.0,
        "min": -65504.0,
        "smallest_normal": 0.00006103515625,
    },
    tf.bfloat16: {
        "bits": 16,
        "eps": 0.0078125,
        "max": 3.3895313892515355e38,
        "min": -3.3895313892515355e38,
        "smallest_normal": 1.1754943508222875e-38,
    },
    tf.float32: {
        "bits": 32,
        "eps": 1.1920928955078125e-07,
        "max": 3.4028234663852886e38,
        "min": -3.4028234663852886e38,
        "smallest_normal": 1.1754943508222875e-38,
    },
    tf.float64: {
        "bits": 64,
        "eps": 2.220446049250313e-16,
        "max": 1.7976931348623157e308,
        "min": -1.7976931348623157e308,
        "smallest_normal": 2.2250738585072014e-308,
    },
}

_iinfo = {
    tf.int8: {"bits": 8, "min": -128, "max": 127},
    tf.int16: {"bits": 16, "min": -32768, "max": 32767},
    tf.int32: {"bits": 32, "min": -2147483648, "max": 2147483647},
    tf.int64: {"bits": 64, "min": -9223372036854775808, "max": 9223372036854775807},
    tf.uint8: {"bits": 8, "min": 0, "max": 255},
    tf.uint16: {"bits": 16, "min": 0, "max": 65535},
    tf.uint32: {"bits": 32, "min": 0, "max": 4294967295},
    tf.uint64: {"bits": 64, "min": 0, "max": 18446744073709551615},
}


def _device_context(device: Device | None):
    return tf.device(device) if device is not None else nullcontext()


def _dlpack_module() -> Any | None:
    dlpack = getattr(tf, "dlpack", None)
    if dlpack is not None:
        return dlpack
    experimental = getattr(tf, "experimental", None)
    return getattr(experimental, "dlpack", None)


def _same_device(requested: Device | None, actual: Device) -> py_bool:
    if requested is None:
        return True
    return requested == actual or str(requested).endswith(str(actual))


def _unwrap(value: Any) -> Any:
    if isinstance(value, Array):
        return value.unwrap()
    return value


def _unwrap_nested(value: Any) -> Any:
    if isinstance(value, Array):
        return value.unwrap()
    if isinstance(value, tuple):
        return tuple(_unwrap_nested(item) for item in value)
    if isinstance(value, list):
        return [_unwrap_nested(item) for item in value]
    return value


def _wrap(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return Array._from_tensor(value)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_wrap(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _python_scalar_dtype(x: complex) -> DType | None:
    if isinstance(x, py_bool):
        return tf.bool
    if isinstance(x, int):
        if _iinfo[tf.int32]["min"] <= x <= _iinfo[tf.int32]["max"]:
            return tf.int32
        if _iinfo[tf.int64]["min"] <= x <= _iinfo[tf.int64]["max"]:
            return tf.int64
        if 0 <= x <= _iinfo[tf.uint64]["max"]:
            return tf.uint64
        return tf.int64
    if isinstance(x, float):
        if math.isfinite(x) and py_abs(x) > _finfo[tf.float32]["max"]:
            return tf.float64
        return tf.float32
    if isinstance(x, complex):
        return tf.complex128
    return None


def _real_dtype_for(dtype: DType) -> DType:
    dtype = tf.as_dtype(dtype)
    if dtype == tf.complex64:
        return tf.float32
    if dtype == tf.complex128:
        return tf.float64
    return dtype


def _is_integer(dtype: DType) -> py_bool:
    return dtype in _integral_dtypes


def _is_complex(dtype: DType) -> py_bool:
    return dtype in _complex_floating_dtypes


def _accumulation_dtype(dtype: DType) -> DType:
    if dtype in (tf.uint8, tf.uint16, tf.uint32):
        return tf.uint32
    if dtype == tf.uint64:
        return tf.uint64
    if dtype in (tf.int8, tf.int16, tf.int32):
        return tf.int32
    if dtype == tf.int64:
        return tf.int64
    return dtype


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def _normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    axes = tuple(_normalize_axis(a, ndim) for a in axis)
    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    return axes


def _shape_tuple(x: Array | tf.Tensor) -> tuple[int | tf.Tensor, ...]:
    tensor = _unwrap(x)
    static_shape = tensor.shape.as_list()
    if py_all(dim is not None for dim in static_shape):
        return tuple(static_shape)
    dynamic_shape = tf.shape(tensor)
    return tuple(
        dynamic_shape[ii] if dim is None else dim for ii, dim in enumerate(static_shape)
    )


def _normalize_shape_arg(shape: int | Sequence[Any] | tf.Tensor) -> Any:
    if isinstance(shape, Array):
        return shape.unwrap()
    if isinstance(shape, tf.Tensor):
        return shape
    if isinstance(shape, int):
        return (shape,)
    return tuple(_unwrap(dim) for dim in shape)


def _shape_arg_tensor(
    shape: int | Sequence[Any] | tf.Tensor,
    dtype: DType = tf.int32,
) -> tf.Tensor:
    shape = _normalize_shape_arg(shape)
    if isinstance(shape, tf.Tensor):
        return tf.cast(shape, dtype)
    return tf.stack(
        [
            tf.cast(dim, dtype)
            if isinstance(dim, tf.Tensor)
            else tf.constant(dim, dtype)
            for dim in shape
        ]
    )


def _shape_arg_for_tf(shape: int | Sequence[Any] | tf.Tensor) -> Any:
    shape = _normalize_shape_arg(shape)
    if isinstance(shape, tuple) and py_any(isinstance(dim, tf.Tensor) for dim in shape):
        return _shape_arg_tensor(shape)
    return shape


def _dtype_of(x: Array | tf.Tensor | DType | complex) -> DType:
    if isinstance(x, tf.DType):
        return x
    if isinstance(x, Array):
        return x.dtype
    if isinstance(x, tf.Tensor):
        return x.dtype
    dtype = _python_scalar_dtype(x)
    if dtype is not None:
        return dtype
    return tf.convert_to_tensor(_unwrap_nested(x)).dtype


def _promote_signed_unsigned(signed: DType, unsigned: DType) -> DType:
    signed_bits = _dtype_bits[signed]
    unsigned_bits = _dtype_bits[unsigned]
    for dtype in (tf.int16, tf.int32, tf.int64):
        bits = _dtype_bits[dtype]
        if bits >= signed_bits and bits > unsigned_bits:
            return dtype
    return tf.float64


def _promote_dtypes(dtype1: DType, dtype2: DType) -> DType:
    dtype1 = tf.as_dtype(dtype1)
    dtype2 = tf.as_dtype(dtype2)
    if dtype1 == dtype2:
        return dtype1
    if dtype1 == tf.bool:
        return dtype2
    if dtype2 == tf.bool:
        return dtype1
    if dtype1 in _complex_floating_dtypes or dtype2 in _complex_floating_dtypes:
        bits = py_max(
            _dtype_bits[_real_dtype_for(dtype1)],
            _dtype_bits[_real_dtype_for(dtype2)],
        )
        return _complex_for_bits[bits]
    if dtype1 in _real_floating_dtypes or dtype2 in _real_floating_dtypes:
        bits = py_max(
            _dtype_bits[dtype1] if dtype1 in _real_floating_dtypes else 0,
            _dtype_bits[dtype2] if dtype2 in _real_floating_dtypes else 0,
        )
        return _float_for_bits[py_max(bits, 32)]
    if dtype1 in _signed_dtypes and dtype2 in _signed_dtypes:
        return dtype1 if _dtype_bits[dtype1] >= _dtype_bits[dtype2] else dtype2
    if dtype1 in _unsigned_dtypes and dtype2 in _unsigned_dtypes:
        return dtype1 if _dtype_bits[dtype1] >= _dtype_bits[dtype2] else dtype2
    if dtype1 in _signed_dtypes and dtype2 in _unsigned_dtypes:
        return _promote_signed_unsigned(dtype1, dtype2)
    if dtype1 in _unsigned_dtypes and dtype2 in _signed_dtypes:
        return _promote_signed_unsigned(dtype2, dtype1)
    raise TypeError(f"Cannot promote {dtype1!r} and {dtype2!r}")


def _promote_scalar(dtype: DType, scalar: complex) -> DType:
    if isinstance(scalar, py_bool):
        return dtype if dtype != tf.bool else tf.bool
    if isinstance(scalar, int):
        return dtype if dtype in _numeric_dtypes else _promote_dtypes(dtype, tf.int32)
    if isinstance(scalar, float):
        if dtype in _real_floating_dtypes | _complex_floating_dtypes:
            return dtype
        return _promote_dtypes(dtype, tf.float64)
    if isinstance(scalar, complex):
        if dtype in _complex_floating_dtypes:
            return dtype
        return _promote_dtypes(dtype, tf.complex128)
    return _promote_dtypes(dtype, _dtype_of(scalar))


def _iter_nested_scalars(obj: Any):
    obj = _unwrap(obj)
    if isinstance(obj, tf.Tensor):
        return
    if isinstance(obj, Sequence) and not isinstance(
        obj, str | bytes | bytearray | memoryview
    ):
        for item in obj:
            yield from _iter_nested_scalars(item)
    else:
        yield obj


def _infer_nested_dtype(obj: Any) -> DType | None:
    dtypes = []
    for scalar in _iter_nested_scalars(obj):
        if scalar is None:
            return None
        dtypes.append(_dtype_of(scalar))
    if not dtypes:
        return None
    return _reduce(_promote_dtypes, dtypes)


def _coerce_scalar_to_dtype(obj: Any, dtype: DType) -> Any:
    if dtype == tf.bool:
        return py_bool(obj)
    if dtype in _integral_dtypes:
        return int(obj)
    if dtype in _real_floating_dtypes:
        return float(obj)
    if dtype in _complex_floating_dtypes:
        return complex(obj)
    return obj


def _coerce_nested_to_dtype(obj: Any, dtype: DType) -> Any:
    obj = _unwrap(obj)
    if isinstance(obj, tf.Tensor):
        return obj
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        try:
            return tf.convert_to_tensor(obj, dtype=dtype)
        except (TypeError, ValueError):
            pass
    if isinstance(obj, Sequence) and not isinstance(
        obj, str | bytes | bytearray | memoryview
    ):
        return [_coerce_nested_to_dtype(item, dtype) for item in obj]
    return _coerce_scalar_to_dtype(obj, dtype)


def _negative_zero(dtype: DType) -> tf.Tensor:
    dtype = tf.as_dtype(dtype)
    if dtype == tf.float16:
        return tf.bitcast(tf.constant(0x8000, dtype=tf.uint16), tf.float16)
    if dtype == tf.bfloat16:
        return tf.bitcast(tf.constant(0x8000, dtype=tf.uint16), tf.bfloat16)
    if dtype == tf.float32:
        return tf.bitcast(tf.constant(0x80000000, dtype=tf.uint32), tf.float32)
    if dtype == tf.float64:
        return tf.bitcast(tf.constant(0x8000000000000000, dtype=tf.uint64), tf.float64)
    raise TypeError(f"{dtype!r} is not a real floating dtype")


def _python_scalar_to_tensor(x: Any, dtype: DType | None) -> tf.Tensor | None:
    if dtype is None:
        dtype = _python_scalar_dtype(x)
    if dtype is None:
        return None
    dtype = tf.as_dtype(dtype)
    if isinstance(x, float) and dtype in _real_floating_dtypes and x == 0.0:
        if math.copysign(1.0, x) < 0:
            return _negative_zero(dtype)
        return tf.zeros((), dtype=dtype)
    if isinstance(x, complex) and dtype in _complex_floating_dtypes:
        real_dtype = _real_dtype_for(dtype)
        real_part = _python_scalar_to_tensor(x.real, real_dtype)
        imag_part = _python_scalar_to_tensor(x.imag, real_dtype)
        if real_part is None:
            real_part = tf.convert_to_tensor(x.real, dtype=real_dtype)
        if imag_part is None:
            imag_part = tf.convert_to_tensor(x.imag, dtype=real_dtype)
        return tf.complex(real_part, imag_part)
    return None


def _astype_tensor(x: tf.Tensor, dtype: DType, copy: py_bool = False) -> tf.Tensor:
    if x.dtype == dtype:
        return tf.identity(x) if copy else x
    return tf.cast(x, dtype)


def _to_tensor(x: Array | tf.Tensor | complex, dtype: DType | None = None) -> tf.Tensor:
    x = _unwrap(x)
    if isinstance(x, tf.Tensor):
        return _astype_tensor(x, dtype) if dtype is not None else x
    out = _python_scalar_to_tensor(x, dtype)
    if out is not None:
        return out
    return tf.convert_to_tensor(_unwrap_nested(x), dtype=dtype)


def result_type(*arrays_and_dtypes: Array | tf.Tensor | DType | complex) -> DType:
    if not arrays_and_dtypes:
        raise ValueError("At least one array or dtype must be provided")
    if py_all(isinstance(x, _py_scalars) for x in arrays_and_dtypes):
        raise ValueError("At least one array or dtype must be provided")
    scalars = []
    others = []
    for x in arrays_and_dtypes:
        if isinstance(x, _py_scalars):
            scalars.append(x)
        else:
            others.append(x)
    dtype = _dtype_of(others[0])
    for other in others[1:]:
        dtype = _promote_dtypes(dtype, _dtype_of(other))
    for scalar in scalars:
        dtype = _promote_scalar(dtype, scalar)
    return dtype


def _result_type_with_scalars(x1: Any, x2: Any) -> DType:
    x1_is_scalar = isinstance(x1, _py_scalars)
    x2_is_scalar = isinstance(x2, _py_scalars)
    if x1_is_scalar and x2_is_scalar:
        return _promote_dtypes(_dtype_of(x1), _dtype_of(x2))
    if x1_is_scalar:
        return _promote_scalar(_dtype_of(x2), x1)
    if x2_is_scalar:
        return _promote_scalar(_dtype_of(x1), x2)
    return result_type(x1, x2)


def _known_unequal(x1: int | None, x2: int | None) -> py_bool:
    return x1 is not None and x2 is not None and x1 != x2


def _promote_two(
    x1: Array | tf.Tensor | complex, x2: Array | tf.Tensor | complex
) -> tuple[tf.Tensor, tf.Tensor]:
    dtype = result_type(x1, x2)
    return _to_tensor(x1, dtype), _to_tensor(x2, dtype)


def _two_arg(f):
    @_wraps(f)
    def _f(x1: Any, x2: Any, /, **kwargs: Any) -> Array:
        x1, x2 = _promote_two(x1, x2)
        return Array._from_tensor(f(x1, x2, **kwargs))

    return _f


def _logical_two_arg(f):
    @_wraps(f)
    def _f(x1: Any, x2: Any, /, **kwargs: Any) -> Array:
        return Array._from_tensor(
            f(
                tf.cast(_unwrap(x1), tf.bool),
                tf.cast(_unwrap(x2), tf.bool),
                **kwargs,
            )
        )

    return _f


def _unary(f):
    @_wraps(f)
    def _f(x: Array, /, **kwargs: Any) -> Array:
        return Array._from_tensor(f(_unwrap(x), **kwargs))

    return _f


def _signed_zero_like(x: tf.Tensor) -> tf.Tensor:
    neg_zero = tf.broadcast_to(_negative_zero(x.dtype), tf.shape(x))
    return tf.where(signbit(Array._from_tensor(x)).unwrap(), neg_zero, tf.zeros_like(x))


def _moveaxis_permutation(
    ndim: int,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> list[int]:
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    if len(source) != len(destination):
        raise ValueError("`source` and `destination` must have the same number of axes")
    source_ = tuple(_normalize_axis(axis, ndim) for axis in source)
    destination_ = tuple(_normalize_axis(axis, ndim) for axis in destination)
    if len(set(source_)) != len(source_) or len(set(destination_)) != len(destination_):
        raise ValueError("repeated axis")
    order = [axis for axis in range(ndim) if axis not in source_]
    for dest, src in sorted(zip(destination_, source_, strict=True)):
        order.insert(dest, src)
    return order


def _moveaxis(
    x: Array | tf.Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> tf.Tensor:
    tensor = _unwrap(x)
    return tf.transpose(
        tensor, _moveaxis_permutation(tensor.shape.rank, source, destination)
    )


def asarray(
    obj: Any,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: py_bool | None = None,
    **kwargs: Any,
) -> Array:
    if copy is False and not isinstance(obj, Array | tf.Tensor):
        raise ValueError("Unable to avoid copy while creating a TensorFlow tensor")
    with _device_context(device):
        if isinstance(obj, Array):
            same_dtype = dtype is None or obj.dtype == dtype
            if copy is False:
                if not same_dtype or not _same_device(device, obj.device):
                    raise ValueError("Unable to avoid copy while converting an Array")
                return obj
            tensor = obj.unwrap() if same_dtype else tf.cast(obj.unwrap(), dtype)
            if device is not None or copy is True:
                tensor = tf.identity(tensor)
            return Array._from_tensor(tensor)
        if isinstance(obj, tf.Tensor):
            tensor = _unwrap(obj)
            same_dtype = dtype is None or tensor.dtype == dtype
            if copy is False and not same_dtype:
                raise ValueError("Unable to avoid copy while converting dtype")
            out = tensor if same_dtype else tf.cast(tensor, dtype)
            if device is not None or copy is True:
                out = tf.identity(out)
            return Array._from_tensor(out)
        try:
            if dtype is None:
                dtype = _infer_nested_dtype(obj)
            out = _python_scalar_to_tensor(obj, dtype)
            if out is None:
                obj_ = (
                    _coerce_nested_to_dtype(obj, dtype)
                    if dtype is not None
                    else _unwrap_nested(obj)
                )
                out = tf.convert_to_tensor(obj_, dtype=dtype, **kwargs)
        except (TypeError, ValueError):
            obj_ = list(obj)
            if dtype is not None:
                obj_ = _coerce_nested_to_dtype(obj_, dtype)
            out = tf.convert_to_tensor(obj_, dtype=dtype, **kwargs)
        return Array._from_tensor(tf.identity(out) if copy is True else out)


def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: py_bool = True,
    device: Device | None = None,
) -> Array:
    with _device_context(device):
        return Array._from_tensor(
            _astype_tensor(_unwrap(x), dtype, copy=copy or device is not None)
        )


def from_dlpack(
    x: Any,
    /,
    *,
    device: Device | None = None,
    copy: py_bool | None = None,
) -> Array:
    if isinstance(x, Array | tf.Tensor):
        return asarray(x, device=device, copy=copy)
    dlpack = _dlpack_module()
    if dlpack is None:
        raise BufferError("TensorFlow DLPack import is not available")
    capsule = x.__dlpack__() if hasattr(x, "__dlpack__") else x
    with _device_context(device):
        out = dlpack.from_dlpack(capsule)
        if device is not None or copy is True:
            out = tf.identity(out)
        return Array._from_tensor(out)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if stop is None:
        start, stop = 0, start
    if (
        isinstance(_unwrap(start), tf.Tensor)
        or isinstance(_unwrap(stop), tf.Tensor)
        or isinstance(_unwrap(step), tf.Tensor)
    ):
        if dtype is None:
            dtype = _dtype_of(stop)
            if dtype not in _numeric_dtypes:
                dtype = tf.int32
        start_ = tf.cast(_to_tensor(start), dtype)
        stop_ = tf.cast(_to_tensor(stop), dtype)
        step_ = tf.cast(_to_tensor(step), dtype)
        with _device_context(device):
            return Array._from_tensor(tf.range(start_, stop_, step_))
    with _device_context(device):
        if step > 0 and stop <= start or step < 0 and stop >= start:
            if dtype is None:
                dtype = (
                    tf.int32
                    if py_all(isinstance(i, int) for i in (start, stop, step))
                    else tf.float32
                )
            return Array._from_tensor(tf.zeros((0,), dtype=dtype))
        if dtype is None:
            if py_all(isinstance(i, int) for i in (start, stop, step)):
                return Array._from_tensor(tf.range(start, stop, step))
            return Array._from_tensor(
                tf.cast(tf.range(start, stop, step, dtype=tf.float64), tf.float32)
            )
        work_dtype = tf.int64 if dtype in _integral_dtypes else tf.float64
        return Array._from_tensor(
            tf.cast(tf.range(start, stop, step, dtype=work_dtype), dtype)
        )


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if isinstance(shape, int):
        shape = (shape,)
    with _device_context(device):
        return Array._from_tensor(
            tf.zeros(_shape_arg_for_tf(shape), dtype=dtype or tf.float32)
        )


def empty_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with _device_context(device):
        return Array._from_tensor(tf.zeros_like(_unwrap(x), dtype=dtype))


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if n_cols is None:
        n_cols = n_rows
    with _device_context(device):
        if k >= n_cols or k <= -n_rows:
            return Array._from_tensor(
                tf.zeros((n_rows, n_cols), dtype=dtype or tf.float32)
            )
        rows = tf.range(n_rows)[:, newaxis]
        cols = tf.range(n_cols)[newaxis, :]
        return Array._from_tensor(tf.cast(cols - rows == k, dtype or tf.float32))


def full(
    shape: int | tuple[int, ...],
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if isinstance(shape, int):
        shape = (shape,)
    with _device_context(device):
        value = _to_tensor(fill_value, dtype=dtype)
        return Array._from_tensor(tf.broadcast_to(value, _shape_arg_for_tf(shape)))


def full_like(
    x: Array,
    /,
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    return full(_shape_tuple(x), fill_value, dtype=dtype or x.dtype, device=device)


def linspace(
    start: float,
    stop: float,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: py_bool = True,
    **kwargs: object,
) -> Array:
    del kwargs
    with _device_context(device):
        if num == 0:
            return Array._from_tensor(tf.zeros((0,), dtype=dtype or tf.float32))
        out_dtype = dtype or tf.float32
        work_dtype = (
            out_dtype
            if out_dtype in _real_floating_dtypes | _complex_floating_dtypes
            else tf.float32
        )
        start_ = tf.convert_to_tensor(start, dtype=work_dtype)
        stop_ = tf.convert_to_tensor(stop, dtype=work_dtype)
        out = tf.linspace(start_, stop_, num if endpoint else num + 1)
        if not endpoint:
            out = out[:-1]
        return Array._from_tensor(tf.cast(out, out_dtype))


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with _device_context(device):
        return Array._from_tensor(
            tf.ones(_shape_arg_for_tf(shape), dtype=dtype or tf.float32)
        )


def ones_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with _device_context(device):
        return Array._from_tensor(tf.ones_like(_unwrap(x), dtype=dtype))


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with _device_context(device):
        return Array._from_tensor(
            tf.zeros(_shape_arg_for_tf(shape), dtype=dtype or tf.float32)
        )


def zeros_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with _device_context(device):
        return Array._from_tensor(tf.zeros_like(_unwrap(x), dtype=dtype))


def tril(x: Array, /, *, k: int = 0) -> Array:
    tensor = _unwrap(x)
    rows = tf.range(tensor.shape[-2])[:, newaxis]
    cols = tf.range(tensor.shape[-1])[newaxis, :]
    return Array._from_tensor(
        tf.where(cols - rows <= k, tensor, tf.zeros((), dtype=tensor.dtype))
    )


def triu(x: Array, /, *, k: int = 0) -> Array:
    tensor = _unwrap(x)
    rows = tf.range(tensor.shape[-2])[:, newaxis]
    cols = tf.range(tensor.shape[-1])[newaxis, :]
    return Array._from_tensor(
        tf.where(cols - rows >= k, tensor, tf.zeros((), dtype=tensor.dtype))
    )


def can_cast(from_: DType | Array, to: DType, /) -> py_bool:
    from_dtype = _dtype_of(from_)
    to = tf.as_dtype(to)
    if from_dtype == to:
        return True
    if from_dtype == tf.bool:
        return to == tf.bool
    if from_dtype in _signed_dtypes:
        if to in _signed_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to]
        if to in _real_floating_dtypes:
            return _dtype_bits[to] > _dtype_bits[from_dtype]
        if to in _complex_floating_dtypes:
            return _dtype_bits[to] // 2 > _dtype_bits[from_dtype]
        return False
    if from_dtype in _unsigned_dtypes:
        if to in _unsigned_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to]
        if to in _signed_dtypes:
            return _dtype_bits[from_dtype] < _dtype_bits[to]
        if to in _real_floating_dtypes:
            return _dtype_bits[to] > _dtype_bits[from_dtype]
        if to in _complex_floating_dtypes:
            return _dtype_bits[to] // 2 > _dtype_bits[from_dtype]
        return False
    if from_dtype in _real_floating_dtypes:
        if to in _real_floating_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to]
        if to in _complex_floating_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to] // 2
        return False
    return (
        from_dtype in _complex_floating_dtypes
        and to in _complex_floating_dtypes
        and _dtype_bits[from_dtype] <= _dtype_bits[to]
    )


def isdtype(
    dtype: DType,
    kind: DType | str | tuple[DType | str, ...],
    *,
    _tuple: py_bool = True,
) -> py_bool:
    dtype = tf.as_dtype(dtype)
    if isinstance(kind, tuple) and _tuple:
        return py_any(isdtype(dtype, k, _tuple=False) for k in kind)
    if isinstance(kind, str):
        if kind == "bool":
            return dtype in _bool_dtypes
        if kind == "signed integer":
            return dtype in _signed_dtypes
        if kind == "unsigned integer":
            return dtype in _unsigned_dtypes
        if kind == "integral":
            return dtype in _integral_dtypes
        if kind == "real floating":
            return dtype in _real_floating_dtypes
        if kind == "complex floating":
            return dtype in _complex_floating_dtypes
        if kind == "numeric":
            return dtype in _numeric_dtypes
        raise ValueError(f"Unrecognized data type kind: {kind!r}")
    return dtype == tf.as_dtype(kind)


class _FInfo:
    def __init__(self, dtype: DType):
        real_dtype = _real_dtype_for(dtype)
        info = _finfo[real_dtype]
        self.bits = info["bits"]
        self.eps = info["eps"]
        self.max = info["max"]
        self.min = info["min"]
        self.smallest_normal = info["smallest_normal"]
        self.dtype = real_dtype


class _IInfo:
    def __init__(self, dtype: DType):
        info = _iinfo[dtype]
        self.bits = info["bits"]
        self.max = info["max"]
        self.min = info["min"]
        self.dtype = dtype


def finfo(type_: DType | Array, /) -> _FInfo:
    return _FInfo(_dtype_of(type_))


def iinfo(type_: DType | Array, /) -> _IInfo:
    return _IInfo(_dtype_of(type_))


def abs(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if tensor.dtype in _unsigned_dtypes or tensor.dtype == tf.bool:
        return Array._from_tensor(tf.identity(tensor))
    return Array._from_tensor(tf.abs(tensor))


acos = _unary(tf.acos)
acosh = _unary(tf.acosh)
asin = _unary(tf.asin)
asinh = _unary(tf.asinh)
atan = _unary(tf.atan)
atan2 = _two_arg(tf.atan2)
atanh = _unary(tf.atanh)
add = _two_arg(tf.add)
conj = _unary(tf.math.conj)
cos = _unary(tf.cos)
cosh = _unary(tf.cosh)
divide = _two_arg(tf.divide)
equal = _two_arg(tf.equal)
exp = _unary(tf.exp)
greater = _two_arg(tf.greater)
greater_equal = _two_arg(tf.greater_equal)
less = _two_arg(tf.less)
less_equal = _two_arg(tf.less_equal)
logical_and = _logical_two_arg(tf.logical_and)
logical_not = _unary(tf.logical_not)
logical_or = _logical_two_arg(tf.logical_or)
logical_xor = _logical_two_arg(tf.math.logical_xor)
maximum = _two_arg(tf.maximum)
minimum = _two_arg(tf.minimum)
multiply = _two_arg(tf.multiply)
not_equal = _two_arg(tf.not_equal)
positive = _unary(tf.identity)
sin = _unary(tf.sin)
sinh = _unary(tf.sinh)
square = _unary(tf.square)
sqrt = _unary(tf.sqrt)
subtract = _two_arg(tf.subtract)
tan = _unary(tf.tan)


def expm1(x: Array, /) -> Array:
    tensor = _unwrap(x)
    out = tf.math.expm1(tensor)
    if _is_complex(tensor.dtype):
        real_part = tf.math.real(tensor)
        imag_part = tf.math.imag(tensor)

        plus_inf_real = tf.math.is_inf(real_part) & (real_part > 0)
        exp_out = tf.exp(tensor) - tf.cast(1, tensor.dtype)
        inf_real = tf.fill(tf.shape(real_part), tf.cast(math.inf, real_part.dtype))
        zero_imag_out = tf.complex(inf_real, imag_part)
        out = tf.where(
            plus_inf_real, tf.where(imag_part == 0, zero_imag_out, exp_out), out
        )

        minus_inf_real = tf.math.is_inf(real_part) & (real_part < 0)
        minus_inf_imag = tf.where(
            tf.math.is_nan(imag_part),
            tf.zeros_like(imag_part),
            _signed_zero_like(imag_part),
        )
        out = tf.where(
            minus_inf_real, tf.complex(-tf.ones_like(real_part), minus_inf_imag), out
        )

        nan_real_zero_imag = tf.math.is_nan(real_part) & (imag_part == 0)
        out = tf.where(nan_real_zero_imag, tf.complex(real_part, imag_part), out)

        zero = tf.zeros((), dtype=real_part.dtype)
        zero_out = tf.complex(tf.zeros_like(real_part), imag_part)
        out = tf.where((real_part == zero) & (imag_part == zero), zero_out, out)
    return Array._from_tensor(out)


def tanh(x: Array, /) -> Array:
    tensor = _unwrap(x)
    out = tf.math.tanh(tensor)
    if tensor.dtype in _real_floating_dtypes | _complex_floating_dtypes:
        out = tf.where(tensor == tf.zeros((), dtype=tensor.dtype), tensor, out)
    if _is_complex(tensor.dtype):
        real_part = tf.math.real(tensor)
        imag_part = tf.math.imag(tensor)
        inf_real = tf.math.is_inf(real_part)
        real_out = tf.where(
            real_part > 0, tf.ones_like(real_part), -tf.ones_like(real_part)
        )
        imag_out = tf.where(
            tf.math.is_nan(imag_part),
            tf.zeros_like(imag_part),
            _signed_zero_like(imag_part),
        )
        out = tf.where(inf_real, tf.complex(real_out, imag_out), out)
    return Array._from_tensor(out)


def remainder(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    out = tf.math.floormod(x1, x2)
    if out.dtype in _real_floating_dtypes:
        signed_zero = tf.broadcast_to(_signed_zero_like(x2), tf.shape(out))
        out = tf.where(out == 0, signed_zero, out)
    return Array._from_tensor(out)


def bitwise_and(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.dtype == tf.bool:
        return Array._from_tensor(tf.logical_and(x1, x2))
    return Array._from_tensor(tf.bitwise.bitwise_and(x1, x2))


def bitwise_left_shift(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    out = tf.bitwise.left_shift(x1, x2)
    return Array._from_tensor(
        tf.where(
            x2 >= tf.cast(_dtype_bits[x1.dtype], x2.dtype),
            tf.zeros((), dtype=x1.dtype),
            out,
        )
    )


def bitwise_invert(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if tensor.dtype == tf.bool:
        return Array._from_tensor(tf.logical_not(tensor))
    return Array._from_tensor(tf.bitwise.invert(tensor))


def bitwise_or(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.dtype == tf.bool:
        return Array._from_tensor(tf.logical_or(x1, x2))
    return Array._from_tensor(tf.bitwise.bitwise_or(x1, x2))


def bitwise_right_shift(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return Array._from_tensor(tf.bitwise.right_shift(x1, x2))


def bitwise_xor(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.dtype == tf.bool:
        return Array._from_tensor(tf.math.logical_xor(x1, x2))
    return Array._from_tensor(tf.bitwise.bitwise_xor(x1, x2))


def ceil(x: Array, /) -> Array:
    tensor = _unwrap(x)
    return Array._from_tensor(
        tf.identity(tensor) if _is_integer(tensor.dtype) else tf.math.ceil(tensor)
    )


def floor(x: Array, /) -> Array:
    tensor = _unwrap(x)
    return Array._from_tensor(
        tf.identity(tensor) if _is_integer(tensor.dtype) else tf.math.floor(tensor)
    )


def trunc(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if _is_integer(tensor.dtype):
        return Array._from_tensor(tf.identity(tensor))
    return Array._from_tensor(
        tf.where(tensor < 0, tf.math.ceil(tensor), tf.math.floor(tensor))
    )


def copysign(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return Array._from_tensor(
        tf.where(signbit(Array._from_tensor(x2)).unwrap(), -tf.abs(x1), tf.abs(x1))
    )


def hypot(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return Array._from_tensor(tf.sqrt(tf.square(x1) + tf.square(x2)))


def imag(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if _is_complex(tensor.dtype):
        return Array._from_tensor(tf.math.imag(tensor))
    return Array._from_tensor(tf.zeros_like(tensor))


def isfinite(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if _is_integer(tensor.dtype) or tensor.dtype == tf.bool:
        return Array._from_tensor(tf.ones(_shape_tuple(tensor), dtype=tf.bool))
    if _is_complex(tensor.dtype):
        return Array._from_tensor(
            tf.math.is_finite(tf.math.real(tensor))
            & tf.math.is_finite(tf.math.imag(tensor))
        )
    return Array._from_tensor(tf.math.is_finite(tensor))


def isinf(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if _is_integer(tensor.dtype) or tensor.dtype == tf.bool:
        return Array._from_tensor(tf.zeros(_shape_tuple(tensor), dtype=tf.bool))
    if _is_complex(tensor.dtype):
        return Array._from_tensor(
            tf.math.is_inf(tf.math.real(tensor)) | tf.math.is_inf(tf.math.imag(tensor))
        )
    return Array._from_tensor(tf.math.is_inf(tensor))


def isnan(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if _is_integer(tensor.dtype) or tensor.dtype == tf.bool:
        return Array._from_tensor(tf.zeros(_shape_tuple(tensor), dtype=tf.bool))
    if _is_complex(tensor.dtype):
        return Array._from_tensor(
            tf.math.is_nan(tf.math.real(tensor)) | tf.math.is_nan(tf.math.imag(tensor))
        )
    return Array._from_tensor(tf.math.is_nan(tensor))


def _complex_log(x: tf.Tensor) -> tf.Tensor:
    return tf.complex(
        tf.math.log(tf.abs(x)), tf.atan2(tf.math.imag(x), tf.math.real(x))
    )


def log(x: Array, /) -> Array:
    tensor = _unwrap(x)
    return Array._from_tensor(
        _complex_log(tensor) if _is_complex(tensor.dtype) else tf.math.log(tensor)
    )


def floor_divide(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return Array._from_tensor(tf.math.floordiv(x1, x2))


def log1p(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if _is_complex(tensor.dtype):
        return log(Array._from_tensor(tf.cast(1, tensor.dtype) + tensor))
    return Array._from_tensor(tf.math.log1p(tensor))


def log2(x: Array, /) -> Array:
    tensor = _unwrap(x)
    out = _complex_log(tensor) if _is_complex(tensor.dtype) else tf.math.log(tensor)
    return Array._from_tensor(out / tf.cast(math.log(2.0), tensor.dtype))


def log10(x: Array, /) -> Array:
    tensor = _unwrap(x)
    out = _complex_log(tensor) if _is_complex(tensor.dtype) else tf.math.log(tensor)
    return Array._from_tensor(out / tf.cast(math.log(10.0), tensor.dtype))


def negative(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if tensor.dtype in _unsigned_dtypes:
        return Array._from_tensor(tf.cast(-tf.cast(tensor, tf.int64), tensor.dtype))
    return Array._from_tensor(tf.negative(tensor))


def pow(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    out_dtype = x1.dtype
    work_dtype = tf.int64 if out_dtype in _integral_dtypes else out_dtype
    return Array._from_tensor(
        tf.cast(tf.pow(tf.cast(x1, work_dtype), tf.cast(x2, work_dtype)), out_dtype)
    )


def logaddexp(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    shape = tuple(tf.broadcast_static_shape(x1.shape, x2.shape).as_list())
    x1 = tf.broadcast_to(x1, shape)
    x2 = tf.broadcast_to(x2, shape)
    return Array._from_tensor(tf.reduce_logsumexp(tf.stack([x1, x2]), axis=0))


def nextafter(x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return Array._from_tensor(tf.math.nextafter(x1, x2))


def real(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if _is_complex(tensor.dtype):
        return Array._from_tensor(tf.math.real(tensor))
    return Array._from_tensor(tf.identity(tensor))


def reciprocal(x: Array, /) -> Array:
    tensor = _unwrap(x)
    return Array._from_tensor(tf.math.reciprocal(tensor))


def round(x: Array, /, *, decimals: int = 0) -> Array:
    tensor = _unwrap(x)
    if tensor.dtype in _integral_dtypes:
        return Array._from_tensor(tf.identity(tensor))
    if _is_complex(tensor.dtype):
        real_part = round(
            Array._from_tensor(tf.math.real(tensor)), decimals=decimals
        ).unwrap()
        imag_part = round(
            Array._from_tensor(tf.math.imag(tensor)), decimals=decimals
        ).unwrap()
        return Array._from_tensor(tf.complex(real_part, imag_part))
    if decimals == 0:
        return Array._from_tensor(tf.round(tensor))
    factor = tf.cast(10**decimals, tensor.dtype)
    return Array._from_tensor(tf.round(tensor * factor) / factor)


def sign(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if tensor.dtype in _unsigned_dtypes:
        return Array._from_tensor(
            tf.where(
                tensor == 0,
                tf.zeros((), dtype=tensor.dtype),
                tf.ones((), dtype=tensor.dtype),
            )
        )
    return Array._from_tensor(tf.sign(tensor))


def signbit(x: Array, /) -> Array:
    tensor = _unwrap(x)
    if tensor.dtype in _unsigned_dtypes or tensor.dtype == tf.bool:
        return Array._from_tensor(tf.zeros(_shape_tuple(tensor), dtype=tf.bool))
    if tensor.dtype in _integral_dtypes:
        return Array._from_tensor(tensor < 0)
    if tensor.dtype == tf.float16:
        return Array._from_tensor(
            tf.bitwise.right_shift(tf.bitcast(tensor, tf.uint16), 15) == 1
        )
    if tensor.dtype == tf.bfloat16:
        return Array._from_tensor(
            tf.bitwise.right_shift(tf.bitcast(tensor, tf.uint16), 15) == 1
        )
    if tensor.dtype == tf.float32:
        return Array._from_tensor(
            tf.bitwise.right_shift(tf.bitcast(tensor, tf.uint32), 31) == 1
        )
    if tensor.dtype == tf.float64:
        return Array._from_tensor(
            tf.bitwise.right_shift(tf.bitcast(tensor, tf.uint64), 63) == 1
        )
    raise TypeError("signbit is only defined for real-valued dtypes")


def clip(
    x: Array,
    /,
    min: Array | complex | None = None,
    max: Array | complex | None = None,
) -> Array:
    tensor = _unwrap(x)
    if min is None and max is None:
        return Array._from_tensor(tf.identity(tensor))
    if min is None:
        max_ = _to_tensor(max, tensor.dtype)
        return Array._from_tensor(tf.minimum(tensor, max_))
    if max is None:
        min_ = _to_tensor(min, tensor.dtype)
        return Array._from_tensor(tf.maximum(tensor, min_))
    min_ = _to_tensor(min, tensor.dtype)
    max_ = _to_tensor(max, tensor.dtype)
    return Array._from_tensor(tf.minimum(tf.maximum(tensor, min_), max_))


def _as_bool(x: Array) -> tf.Tensor:
    tensor = _unwrap(x)
    return (
        tensor
        if tensor.dtype == tf.bool
        else tensor != tf.zeros((), dtype=tensor.dtype)
    )


def all(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    return Array._from_tensor(tf.reduce_all(_as_bool(x), axis=axis, keepdims=keepdims))


def any(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    return Array._from_tensor(tf.reduce_any(_as_bool(x), axis=axis, keepdims=keepdims))


def max(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    return Array._from_tensor(tf.reduce_max(_unwrap(x), axis=axis, keepdims=keepdims))


def min(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    return Array._from_tensor(tf.reduce_min(_unwrap(x), axis=axis, keepdims=keepdims))


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: py_bool = False,
) -> Array:
    tensor = _unwrap(x)
    if dtype is not None:
        tensor = tf.cast(tensor, dtype)
    return Array._from_tensor(tf.reduce_mean(tensor, axis=axis, keepdims=keepdims))


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: py_bool = False,
) -> Array:
    tensor = _unwrap(x)
    dtype = dtype or _accumulation_dtype(tensor.dtype)
    return Array._from_tensor(
        tf.reduce_prod(tf.cast(tensor, dtype), axis=axis, keepdims=keepdims)
    )


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: py_bool = False,
) -> Array:
    tensor = _unwrap(x)
    dtype = dtype or _accumulation_dtype(tensor.dtype)
    return Array._from_tensor(
        tf.reduce_sum(tf.cast(tensor, dtype), axis=axis, keepdims=keepdims)
    )


def _axis_size(x: tf.Tensor, axis: int | tuple[int, ...] | None) -> int:
    if axis is None:
        return math.prod(_shape_tuple(x))
    axes = _normalize_axes(axis, x.shape.rank)
    return math.prod(_shape_tuple(x)[a] for a in axes)


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    dtype: DType | None = None,
    keepdims: py_bool = False,
) -> Array:
    tensor = _unwrap(x)
    dtype = dtype or tensor.dtype
    tensor = tf.cast(tensor, dtype)
    if axis == ():
        return Array._from_tensor(tf.zeros_like(tensor))
    mean_ = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    centered = tensor - mean_
    if _is_complex(tensor.dtype):
        squared = tf.math.real(centered * tf.math.conj(centered))
    else:
        squared = tf.square(centered)
    n = _axis_size(tensor, axis)
    out = tf.reduce_sum(squared, axis=axis, keepdims=keepdims) / tf.cast(
        n - correction,
        squared.dtype,
    )
    return Array._from_tensor(out)


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    dtype: DType | None = None,
    keepdims: py_bool = False,
) -> Array:
    return sqrt(
        var(x, axis=axis, correction=correction, dtype=dtype, keepdims=keepdims)
    )


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: py_bool = False,
) -> Array:
    tensor = _unwrap(x)
    if axis is None:
        if tensor.shape.rank > 1:
            raise ValueError(
                "axis must be specified in cumulative_sum for more than one dimension"
            )
        axis = 0
    axis = _normalize_axis(axis, tensor.shape.rank)
    dtype = dtype or _accumulation_dtype(tensor.dtype)
    tensor = tf.cast(tensor, dtype)
    out = tf.cumsum(tensor, axis=axis)
    if include_initial:
        shape = list(out.shape.as_list())
        shape[axis] = 1
        out = tf.concat([tf.zeros(shape, dtype=out.dtype), out], axis=axis)
    return Array._from_tensor(out)


def cumulative_prod(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: py_bool = False,
) -> Array:
    tensor = _unwrap(x)
    if axis is None:
        if tensor.shape.rank > 1:
            raise ValueError(
                "axis must be specified in cumulative_prod for more than one dimension"
            )
        axis = 0
    axis = _normalize_axis(axis, tensor.shape.rank)
    dtype = dtype or _accumulation_dtype(tensor.dtype)
    tensor = tf.cast(tensor, dtype)
    out = tf.math.cumprod(tensor, axis=axis)
    if include_initial:
        shape = list(out.shape.as_list())
        shape[axis] = 1
        out = tf.concat([tf.ones(shape, dtype=out.dtype), out], axis=axis)
    return Array._from_tensor(out)


def diff(
    x: Array,
    /,
    *,
    axis: int = -1,
    n: int = 1,
    prepend: Array | None = None,
    append: Array | None = None,
) -> Array:
    tensor = _unwrap(x)
    axis = _normalize_axis(axis, tensor.shape.rank)
    parts = []
    if prepend is not None:
        parts.append(_unwrap(prepend))
    parts.append(tensor)
    if append is not None:
        parts.append(_unwrap(append))
    tensor = tf.concat(parts, axis=axis) if len(parts) > 1 else tensor
    for _ in range(n):
        upper = [slice(None)] * tensor.shape.rank
        lower = [slice(None)] * tensor.shape.rank
        upper[axis] = slice(1, None)
        lower[axis] = slice(None, -1)
        tensor = tensor[tuple(upper)] - tensor[tuple(lower)]
    return Array._from_tensor(tensor)


def argsort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    del stable
    return Array._from_tensor(
        tf.argsort(
            _unwrap(x),
            axis=axis,
            direction="DESCENDING" if descending else "ASCENDING",
            stable=True,
        )
    )


def sort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    del stable
    return Array._from_tensor(
        tf.sort(
            _unwrap(x), axis=axis, direction="DESCENDING" if descending else "ASCENDING"
        )
    )


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    tensor = _unwrap(x)
    if axis is None:
        tensor = tf.reshape(tensor, (-1,))
        axis = 0
    axis = _normalize_axis(axis, tensor.shape.rank)
    indices_ = tf.cast(_unwrap(indices), tf.int64)
    dim = tensor.shape[axis]
    if dim is not None:
        indices_ = tf.where(indices_ < 0, indices_ + tf.cast(dim, tf.int64), indices_)
    return Array._from_tensor(tf.gather(tensor, indices_, axis=axis))


def take_along_axis(x: Array, indices: Array, /, *, axis: int = -1) -> Array:
    tensor = _unwrap(x)
    indices_ = tf.cast(_unwrap(indices), tf.int64)
    axis = _normalize_axis(axis, tensor.shape.rank)
    dim = tensor.shape[axis]
    if dim is not None:
        indices_ = tf.where(indices_ < 0, indices_ + tf.cast(dim, tf.int64), indices_)
    out_shape = tf.shape(indices_)
    tensor_shape = _shape_tuple(tensor)
    coords = []
    for dim_axis, dim_size in enumerate(tensor_shape):
        if dim_axis == axis:
            coords.append(indices_)
            continue
        shape = [1] * tensor.shape.rank
        shape[dim_axis] = dim_size
        coord = tf.reshape(tf.range(dim_size, dtype=tf.int64), shape)
        coords.append(tf.broadcast_to(coord, out_shape))
    return Array._from_tensor(tf.gather_nd(tensor, tf.stack(coords, axis=-1)))


def matmul(x1: Array, x2: Array, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if (
        x1.shape.rank == 0
        or x2.shape.rank == 0
        or x1.shape.rank == x2.shape.rank == 1
        and _known_unequal(x1.shape[0], x2.shape[0])
        or x1.shape.rank == 1
        and x2.shape.rank >= 2
        and _known_unequal(x1.shape[0], x2.shape[-2])
        or x2.shape.rank == 1
        and x1.shape.rank >= 2
        and _known_unequal(x2.shape[0], x1.shape[-1])
        or x1.shape.rank >= 2
        and x2.shape.rank >= 2
        and _known_unequal(x1.shape[-1], x2.shape[-2])
    ):
        raise ValueError("matmul input shapes are incompatible")
    out_dtype = x1.dtype
    work_dtype = tf.int64 if out_dtype in _integral_dtypes else out_dtype
    x1 = tf.cast(x1, work_dtype)
    x2 = tf.cast(x2, work_dtype)
    x1_was_vector = x1.shape.rank == 1
    x2_was_vector = x2.shape.rank == 1
    if x1_was_vector:
        x1 = tf.expand_dims(x1, -2)
    if x2_was_vector:
        x2 = tf.expand_dims(x2, -1)
    batch_shape = tf.broadcast_static_shape(x1.shape[:-2], x2.shape[:-2])
    dynamic_batch_shape = tf.broadcast_dynamic_shape(
        tf.shape(x1)[:-2], tf.shape(x2)[:-2]
    )
    x1_matrix_shape = x1.shape[-2:]
    x2_matrix_shape = x2.shape[-2:]
    x1 = tf.broadcast_to(
        x1, tf.concat([dynamic_batch_shape, tf.shape(x1)[-2:]], axis=0)
    )
    x2 = tf.broadcast_to(
        x2, tf.concat([dynamic_batch_shape, tf.shape(x2)[-2:]], axis=0)
    )
    x1.set_shape(batch_shape.concatenate(x1_matrix_shape))
    x2.set_shape(batch_shape.concatenate(x2_matrix_shape))
    out = tf.reduce_sum(tf.expand_dims(x1, -1) * tf.expand_dims(x2, -3), axis=-2)
    if x1_was_vector:
        out = tf.squeeze(out, axis=-2)
    if x2_was_vector:
        out = tf.squeeze(out, axis=-1)
    return Array._from_tensor(tf.cast(out, out_dtype))


def matrix_transpose(x: Array, /) -> Array:
    return Array._from_tensor(tf.linalg.matrix_transpose(_unwrap(x)))


def tensordot(
    x1: Array,
    x2: Array,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if isinstance(axes, int):
        axes1 = tuple(range(x1.shape.rank - axes, x1.shape.rank))
        axes2 = tuple(range(axes))
    else:
        axes1, axes2 = tuple(axes[0]), tuple(axes[1])
    axes1 = tuple(_normalize_axis(axis, x1.shape.rank) for axis in axes1)
    axes2 = tuple(_normalize_axis(axis, x2.shape.rank) for axis in axes2)
    if len(axes1) != len(axes2):
        raise ValueError("tensordot axes must have the same length")
    for axis1, axis2 in zip(axes1, axes2, strict=True):
        if x1.shape[axis1] != x2.shape[axis2]:
            raise ValueError("tensordot contraction dimensions must match")

    x1_outer = tuple(axis for axis in range(x1.shape.rank) if axis not in axes1)
    x2_outer = tuple(axis for axis in range(x2.shape.rank) if axis not in axes2)
    x1_perm = x1_outer + axes1
    x2_perm = axes2 + x2_outer
    x1_t = tf.transpose(x1, x1_perm) if x1_perm else x1
    x2_t = tf.transpose(x2, x2_perm) if x2_perm else x2
    x1_outer_shape = tuple(x1.shape[axis] for axis in x1_outer)
    x2_outer_shape = tuple(x2.shape[axis] for axis in x2_outer)
    contract_shape = tuple(x1.shape[axis] for axis in axes1)
    outer1 = math.prod(x1_outer_shape) if x1_outer_shape else 1
    outer2 = math.prod(x2_outer_shape) if x2_outer_shape else 1
    contract = math.prod(contract_shape) if contract_shape else 1
    x1_m = tf.reshape(x1_t, (outer1, contract))
    x2_m = tf.reshape(x2_t, (contract, outer2))
    out = matmul(Array._from_tensor(x1_m), Array._from_tensor(x2_m)).unwrap()
    return Array._from_tensor(tf.reshape(out, x1_outer_shape + x2_outer_shape))


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    x1, x2 = _promote_two(x1, x2)
    shape = tuple(tf.broadcast_static_shape(x1.shape, x2.shape).as_list())
    axis = _normalize_axis(axis, len(shape))
    x1_shape = (1,) * (len(shape) - x1.shape.rank) + _shape_tuple(x1)
    x2_shape = (1,) * (len(shape) - x2.shape.rank) + _shape_tuple(x2)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError("vecdot contraction dimensions must match")
    x1 = tf.broadcast_to(x1, shape)
    x2 = tf.broadcast_to(x2, shape)
    if _is_complex(x1.dtype):
        x1 = tf.math.conj(x1)
    work_dtype = tf.int64 if x1.dtype in _integral_dtypes else x1.dtype
    out = tf.reduce_sum(tf.cast(x1, work_dtype) * tf.cast(x2, work_dtype), axis=axis)
    return Array._from_tensor(tf.cast(out, x1.dtype))


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    shape = tf.TensorShape(())
    for item in shapes:
        shape = tf.broadcast_static_shape(shape, tf.TensorShape(item))
    return tuple(shape.as_list())


def broadcast_to(x: Array, /, shape: tuple[int, ...]) -> Array:
    return Array._from_tensor(tf.broadcast_to(_unwrap(x), _shape_arg_for_tf(shape)))


def broadcast_arrays(*arrays: Array) -> tuple[Array, ...]:
    shape = broadcast_shapes(*(_shape_tuple(x) for x in arrays))
    return tuple(broadcast_to(x, shape) for x in arrays)


def concat(
    arrays: tuple[Array, ...] | list[Array],
    /,
    *,
    axis: int | None = 0,
) -> Array:
    dtype = result_type(*arrays)
    tensors = [_to_tensor(x, dtype) for x in arrays]
    if axis is None:
        tensors = [tf.reshape(x, (-1,)) for x in tensors]
        axis = 0
    return Array._from_tensor(tf.concat(tensors, axis=axis))


def expand_dims(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    tensor = _unwrap(x)
    if isinstance(axis, int):
        axis = (axis,)
    final_ndim = tensor.shape.rank + len(axis)
    axes = tuple(a + final_ndim if a < 0 else a for a in axis)
    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    if py_any(a < 0 or a >= final_ndim for a in axes):
        raise IndexError("axis out of bounds")
    shape = list(_shape_tuple(tensor))
    for a in sorted(axes):
        shape.insert(a, 1)
    return Array._from_tensor(tf.reshape(tensor, shape))


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    tensor = _unwrap(x)
    axes = _normalize_axes(axis, tensor.shape.rank)
    return Array._from_tensor(tf.reverse(tensor, axes))


def meshgrid(*arrays: Array, indexing: Literal["xy", "ij"] = "xy") -> tuple[Array, ...]:
    return tuple(
        Array._from_tensor(x)
        for x in tf.meshgrid(*[_unwrap(a) for a in arrays], indexing=indexing)
    )


def moveaxis(
    x: Array,
    /,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Array:
    return Array._from_tensor(_moveaxis(x, source, destination))


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    return Array._from_tensor(tf.transpose(_unwrap(x), axes))


def transpose(x: Array, /, axes: tuple[int, ...] | None = None) -> Array:
    tensor = _unwrap(x)
    if axes is None:
        axes = tuple(range(tensor.shape.rank - 1, -1, -1))
    return Array._from_tensor(tf.transpose(tensor, axes))


def einsum(subscripts: str, *operands: Array) -> Array:
    return Array._from_tensor(tf.einsum(subscripts, *[_to_tensor(x) for x in operands]))


def repeat(x: Array, repeats: int | Array, /, *, axis: int | None = None) -> Array:
    tensor = _unwrap(x)
    if axis is None:
        tensor = tf.reshape(tensor, (-1,))
        axis = 0
    axis = _normalize_axis(axis, tensor.shape.rank)
    n = tf.shape(tensor, out_type=tf.int64)[axis]
    repeats_ = _unwrap(repeats)
    if isinstance(repeats_, tf.Tensor):
        repeats_ = tf.cast(repeats_, tf.int64)
    indices = tf.repeat(tf.range(n, dtype=tf.int64), repeats_)
    return Array._from_tensor(tf.gather(tensor, indices, axis=axis))


def reshape(
    x: Array, /, shape: tuple[int, ...], *, copy: py_bool | None = None
) -> Array:
    del copy
    return Array._from_tensor(tf.reshape(_unwrap(x), _shape_arg_for_tf(shape)))


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    tensor = _unwrap(x)
    if axis is None:
        if tensor.shape.rank == 0:
            return Array._from_tensor(tf.identity(tensor))
        shape = _shape_tuple(tensor)
        out = tf.roll(tf.reshape(tensor, (-1,)), shift, 0)
        return Array._from_tensor(tf.reshape(out, shape))
    return Array._from_tensor(tf.roll(tensor, shift, axis))


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    tensor = _unwrap(x)
    axes = _normalize_axes(axis, tensor.shape.rank)
    if axes == ():
        return Array._from_tensor(tf.identity(tensor))
    if py_any(tensor.shape[axis] != 1 for axis in axes):
        raise ValueError("cannot squeeze an axis whose size is not 1")
    return Array._from_tensor(tf.squeeze(tensor, axis=axes))


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    dtype = result_type(*arrays)
    return Array._from_tensor(
        tf.stack([_to_tensor(x, dtype) for x in arrays], axis=axis)
    )


def tile(x: Array, repetitions: tuple[int, ...], /) -> Array:
    tensor = _unwrap(x)
    repetitions = tuple(_unwrap(rep) for rep in repetitions)
    if tensor.shape.rank > len(repetitions):
        repetitions = (1,) * (tensor.shape.rank - len(repetitions)) + tuple(repetitions)
    elif tensor.shape.rank < len(repetitions):
        tensor = tf.reshape(
            tensor, (1,) * (len(repetitions) - tensor.shape.rank) + _shape_tuple(tensor)
        )
    return Array._from_tensor(tf.tile(tensor, _shape_arg_tensor(repetitions)))


def unstack(x: Array, /, *, axis: int = 0) -> tuple[Array, ...]:
    return tuple(Array._from_tensor(item) for item in tf.unstack(_unwrap(x), axis=axis))


def argmax(x: Array, /, *, axis: int | None = None, keepdims: py_bool = False) -> Array:
    tensor = _unwrap(x)
    if axis is None:
        out = tf.argmax(tf.reshape(tensor, (-1,)), axis=0, output_type=tf.int64)
        if keepdims:
            out = tf.reshape(out, (1,) * tensor.shape.rank)
        return Array._from_tensor(out)
    out = tf.argmax(tensor, axis=axis, output_type=tf.int64)
    if keepdims:
        out = tf.expand_dims(out, axis)
    return Array._from_tensor(out)


def argmin(x: Array, /, *, axis: int | None = None, keepdims: py_bool = False) -> Array:
    tensor = _unwrap(x)
    if axis is None:
        out = tf.argmin(tf.reshape(tensor, (-1,)), axis=0, output_type=tf.int64)
        if keepdims:
            out = tf.reshape(out, (1,) * tensor.shape.rank)
        return Array._from_tensor(out)
    out = tf.argmin(tensor, axis=axis, output_type=tf.int64)
    if keepdims:
        out = tf.expand_dims(out, axis)
    return Array._from_tensor(out)


def count_nonzero(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    return Array._from_tensor(
        tf.math.count_nonzero(_unwrap(x), axis=axis, keepdims=keepdims, dtype=tf.int64)
    )


def nonzero(x: Array, /) -> tuple[Array, ...]:
    indices = tf.where(_as_bool(x))
    return tuple(
        Array._from_tensor(item) for item in tf.unstack(tf.transpose(indices), axis=0)
    )


def searchsorted(
    x1: Array,
    x2: Array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Array | None = None,
) -> Array:
    if sorter is not None:
        x1 = take(x1, sorter)
    x1_ = _unwrap(x1)
    x2_ = _to_tensor(x2, x1_.dtype)
    if x1_.shape.rank == 1:
        out = tf.searchsorted(x1_, tf.reshape(x2_, (-1,)), side=side, out_type=tf.int64)
        return Array._from_tensor(tf.reshape(out, _shape_tuple(x2_)))
    return Array._from_tensor(tf.searchsorted(x1_, x2_, side=side, out_type=tf.int64))


def where(condition: Array, x1: Any, x2: Any, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return Array._from_tensor(tf.where(_unwrap(condition), x1, x2))


def _isnan_tensor(x: tf.Tensor) -> tf.Tensor:
    if x.dtype in _real_floating_dtypes:
        return tf.math.is_nan(x)
    if x.dtype in _complex_floating_dtypes:
        return tf.math.is_nan(tf.math.real(x)) | tf.math.is_nan(tf.math.imag(x))
    return tf.zeros(tf.shape(x), dtype=tf.bool)


def _unique(x: Array) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    flat = tf.reshape(_unwrap(x), (-1,))
    n = tf.shape(flat, out_type=tf.int64)[0]
    matrix_shape = tf.stack([n, n])
    idx = tf.range(n, dtype=tf.int64)
    equality = tf.equal(tf.expand_dims(flat, 1), tf.expand_dims(flat, 0))
    nan = _isnan_tensor(flat)
    same_position = tf.equal(tf.expand_dims(idx, 1), tf.expand_dims(idx, 0))
    equality = equality | (
        tf.expand_dims(nan, 1) & tf.expand_dims(nan, 0) & same_position
    )

    first_indices = tf.reduce_min(
        tf.where(
            equality,
            tf.broadcast_to(idx, matrix_shape),
            tf.fill(matrix_shape, tf.cast(n, tf.int64)),
        ),
        axis=1,
    )
    unique_mask = tf.equal(first_indices, idx)
    unique_mask.set_shape((None,))
    indices = tf.boolean_mask(idx, unique_mask)
    values = tf.gather(flat, indices)
    inverse_equality = tf.equal(
        tf.expand_dims(first_indices, 1), tf.expand_dims(indices, 0)
    )
    inverse = tf.argmax(
        tf.cast(inverse_equality, tf.int64), axis=1, output_type=tf.int64
    )
    counts = tf.reduce_sum(tf.cast(inverse_equality, tf.int64), axis=0)
    return values, indices, inverse, counts


def unique_all(x: Array) -> UniqueAllResult:
    values, indices, inverse, counts = _unique(x)
    return UniqueAllResult(
        Array._from_tensor(values),
        Array._from_tensor(indices),
        Array._from_tensor(tf.reshape(inverse, tf.shape(_unwrap(x)))),
        Array._from_tensor(counts),
    )


def unique_counts(x: Array) -> UniqueCountsResult:
    values, _, _, counts = _unique(x)
    return UniqueCountsResult(Array._from_tensor(values), Array._from_tensor(counts))


def unique_inverse(x: Array) -> UniqueInverseResult:
    values, _, inverse, _ = _unique(x)
    return UniqueInverseResult(
        Array._from_tensor(values),
        Array._from_tensor(tf.reshape(inverse, tf.shape(_unwrap(x)))),
    )


def unique_values(x: Array) -> Array:
    values, _, _, _ = _unique(x)
    return Array._from_tensor(values)


def isin(x1: Array | int, x2: Array | int, /, *, invert: py_bool = False) -> Array:
    dtype = _result_type_with_scalars(x1, x2)
    x1_ = _to_tensor(x1, dtype)
    x2_ = tf.reshape(_to_tensor(x2, dtype), (-1,))
    out = tf.reduce_any(tf.equal(tf.expand_dims(x1_, -1), x2_), axis=-1)
    return Array._from_tensor(tf.logical_not(out) if invert else out)


__all__ = [
    "Array",
    "DType",
    "Device",
    "UniqueAllResult",
    "UniqueCountsResult",
    "UniqueInverseResult",
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asarray",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bfloat16",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "bool",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "can_cast",
    "ceil",
    "clip",
    "complex64",
    "complex128",
    "concat",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "count_nonzero",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "divide",
    "e",
    "einsum",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "expand_dims",
    "expm1",
    "eye",
    "finfo",
    "flip",
    "float16",
    "float32",
    "float64",
    "floor",
    "floor_divide",
    "from_dlpack",
    "full",
    "full_like",
    "greater",
    "greater_equal",
    "hypot",
    "iinfo",
    "imag",
    "inf",
    "int8",
    "int16",
    "int32",
    "int64",
    "isdtype",
    "isfinite",
    "isinf",
    "isin",
    "isnan",
    "less",
    "less_equal",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "matrix_transpose",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "moveaxis",
    "multiply",
    "nan",
    "negative",
    "nextafter",
    "newaxis",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "permute_dims",
    "pi",
    "positive",
    "pow",
    "prod",
    "real",
    "reciprocal",
    "remainder",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "round",
    "searchsorted",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sort",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "tril",
    "triu",
    "trunc",
    "transpose",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unstack",
    "var",
    "vecdot",
    "where",
    "zeros",
    "zeros_like",
]


def __dir__() -> list[str]:
    return __all__
