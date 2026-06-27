from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from enum import IntEnum
from typing import Any

import tensorflow as tf
from tensorflow.python.framework import composite_tensor_gradient


class DLDeviceType(IntEnum):
    CPU = 1
    CUDA = 2


class _ArrayGradient(composite_tensor_gradient.CompositeTensorGradient):
    def get_gradient_components(self, value: Array) -> tf.Tensor:
        return value._tensor

    def replace_gradient_components(
        self,
        value: Array,
        component_grads: tf.Tensor | None,
    ) -> Array | None:
        del value
        if component_grads is None:
            return None
        return Array._from_tensor(component_grads)


class Array(tf.experimental.ExtensionType):
    """User-facing TensorFlow-backed array object.

    The object owns the Array API surface. TensorFlow tensors remain plain
    TensorFlow tensors; no TensorFlow class is patched.
    """

    __array_priority__ = 1
    __composite_gradient__ = _ArrayGradient()

    _tensor: tf.Tensor

    def __init__(self, tensor: Any | None = None, /) -> None:
        if tensor is None:
            raise TypeError(
                "'Array' cannot be instantiated without data. Use "
                "'ndtensorflow.asarray' or another creation function instead."
            )
        if not isinstance(tensor, tf.Tensor):
            tensor = tf.convert_to_tensor(tensor)
        self._tensor = tensor

    @classmethod
    def _from_tensor(cls, tensor: Any, /) -> Array:
        if isinstance(tensor, Array):
            return tensor
        return cls(tensor)

    def _replace_tensor(self, tensor: tf.Tensor) -> None:
        self.__dict__["_tensor"] = tensor

    def unwrap(self) -> tf.Tensor:
        """Return the wrapped TensorFlow tensor."""
        return self._tensor

    def __tf_tensor__(
        self,
        dtype: tf.DType | None = None,
        name: str | None = None,
    ) -> tf.Tensor:
        del name
        return tf.cast(self._tensor, dtype) if dtype is not None else self._tensor

    @property
    def device(self) -> str:
        device = self._tensor.device
        marker = "/device:"
        if marker in device:
            return device.rsplit(marker, maxsplit=1)[-1]
        return device

    @property
    def dtype(self) -> tf.DType:
        return self._tensor.dtype

    @property
    def mT(self) -> Array:  # noqa: N802
        from deepmd._vendors import ndtensorflow as xp

        return xp.matrix_transpose(self)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int | tf.Tensor, ...]:
        static_shape = self._tensor.shape.as_list()
        if all(dim is not None for dim in static_shape):
            return tuple(static_shape)
        dynamic_shape = tf.shape(self._tensor)
        return tuple(
            dynamic_shape[ii] if dim is None else dim
            for ii, dim in enumerate(static_shape)
        )

    @property
    def size(self) -> int | tf.Tensor:
        shape = self.shape
        if all(isinstance(dim, int) for dim in shape):
            return math.prod(shape)
        return tf.size(self._tensor)

    @property
    def T(self) -> Array:  # noqa: N802
        from deepmd._vendors import ndtensorflow as xp

        return xp.permute_dims(self, tuple(range(self.ndim - 1, -1, -1)))

    def astype(
        self,
        dtype: tf.DType,
        /,
        *,
        copy: bool = True,
        device: str | None = None,
    ) -> Array:
        from deepmd._vendors import ndtensorflow as xp

        return xp.astype(self, dtype, copy=copy, device=device)

    def reshape(self, *shape: Any, copy: bool | None = None) -> Array:
        from deepmd._vendors import ndtensorflow as xp

        if len(shape) == 1 and isinstance(shape[0], tuple | list):
            shape = tuple(shape[0])
        return xp.reshape(self, tuple(shape), copy=copy)

    def ravel(self) -> Array:
        from deepmd._vendors import ndtensorflow as xp

        return xp.reshape(self, (-1,))

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> Array:
        if axis is None:
            return type(self)._from_tensor(tf.squeeze(self._tensor))
        from deepmd._vendors import ndtensorflow as xp

        return xp.squeeze(self, axis=axis)

    def to_device(self, device: str, /, *, stream: int | Any | None = None) -> Array:
        del stream
        from deepmd._vendors import ndtensorflow as xp

        return xp.asarray(self, device=device, copy=True)

    def __dlpack__(
        self,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        del stream, max_version, dl_device, copy
        if hasattr(self._tensor, "__dlpack__"):
            return self._tensor.__dlpack__()
        raise BufferError(
            "TensorFlow exposes DLPack conversion only outside the stable "
            "Tensor API in this environment."
        )

    def __dlpack_device__(self) -> tuple[DLDeviceType, int]:
        device = self.device.upper()
        if "GPU" in device:
            index = int(device.rsplit(":", maxsplit=1)[-1])
            return (DLDeviceType.CUDA, index)
        return (DLDeviceType.CPU, 0)

    def __array_namespace__(self, /, *, api_version: str | None = None) -> Any:
        del api_version
        from deepmd._vendors import ndtensorflow as xp

        return xp

    def __array__(self, dtype: Any | None = None) -> Any:
        if not tf.executing_eagerly():
            raise TypeError("cannot convert a TensorFlow graph tensor to a NumPy array")
        array = self._tensor.numpy()
        return array.astype(dtype) if dtype is not None else array

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of unsized array")
        dim = self.shape[0]
        if not isinstance(dim, int):
            raise TypeError("len() requires a statically known leading dimension")
        return dim

    def __iter__(self) -> Iterator[Array]:
        if self.ndim == 0:
            raise ValueError("iteration over a 0-d array")
        return (self[i] for i in range(len(self)))

    def __getitem__(self, key: Any, /) -> Array:
        key = _normalize_index_key(key)
        if isinstance(key, tf.Tensor) and key.dtype == tf.bool:
            if key.shape.rank > self.ndim or not all(
                key_dim in (x_dim, 0) for x_dim, key_dim in zip(self.shape, key.shape)
            ):
                raise IndexError(
                    "boolean index shape is incompatible with indexed array"
                )
            if key.shape.rank == 0:
                tensor = tf.expand_dims(self._tensor, 0)
                mask = tf.reshape(key, (1,))
                return type(self)._from_tensor(tf.boolean_mask(tensor, mask))
            if any(dim == 0 for dim in key.shape):
                shape = (0,) + self.shape[key.shape.rank :]
                return type(self)._from_tensor(tf.zeros(shape, dtype=self.dtype))
            return type(self)._from_tensor(tf.boolean_mask(self._tensor, key))
        if (
            isinstance(key, tf.Tensor)
            and _is_integer_index_tensor(key)
            and key.shape.rank != 0
        ):
            return type(self)._from_tensor(
                tf.gather(
                    self._tensor, _normalize_integer_index(key, self.shape[0]), axis=0
                )
            )
        scalar = _scalar_integer_getitem(self._tensor, key)
        if scalar is not None:
            return type(self)._from_tensor(scalar)
        advanced = _advanced_integer_getitem(self._tensor, key)
        if advanced is not None:
            return type(self)._from_tensor(advanced)
        return type(self)._from_tensor(self._tensor[key])

    def __setitem__(self, key: Any, value: Any, /) -> None:
        key = _normalize_index_key(key)
        value_tensor = _value_to_tensor(value, dtype=self.dtype)

        if isinstance(key, tf.Tensor) and key.dtype == tf.bool:
            self._replace_tensor(
                tf.where(
                    key,
                    tf.broadcast_to(value_tensor, tf.shape(self._tensor)),
                    self._tensor,
                )
            )
            return

        variable = tf.Variable(self._tensor)
        try:
            variable[key].assign(value_tensor)
        except Exception as exc:  # pragma: no cover - TensorFlow controls details.
            raise TypeError(
                f"unsupported TensorFlow assignment index: {key!r}"
            ) from exc
        self._replace_tensor(tf.convert_to_tensor(variable))

    def _scalar_value(self) -> Any:
        if self.ndim != 0:
            raise TypeError("only 0-d arrays can be converted to Python scalars")
        if not tf.executing_eagerly():
            raise TypeError(
                "cannot convert a TensorFlow graph tensor to a Python scalar"
            )
        return self._tensor.numpy().item()

    def __bool__(self, /) -> bool:
        return bool(self._scalar_value())

    def __complex__(self, /) -> complex:
        return complex(self._scalar_value())

    def __float__(self, /) -> float:
        return float(self._scalar_value())

    def __index__(self, /) -> int:
        return int(self._scalar_value())

    def __int__(self, /) -> int:
        return int(self._scalar_value())

    def __repr__(self) -> str:
        return f"ndtensorflow.asarray({self._tensor!r})"

    def __eq__(self, other: Any) -> Array:  # type: ignore[override]
        if not _is_supported_operand(other):
            return NotImplemented
        from deepmd._vendors import ndtensorflow as xp

        return xp.equal(self, other)

    def __ne__(self, other: Any) -> Array:  # type: ignore[override]
        if not _is_supported_operand(other):
            return NotImplemented
        from deepmd._vendors import ndtensorflow as xp

        return xp.not_equal(self, other)

    def __abs__(self) -> Array:
        from deepmd._vendors import ndtensorflow as xp

        return xp.abs(self)

    def __invert__(self) -> Array:
        from deepmd._vendors import ndtensorflow as xp

        return xp.bitwise_invert(self)

    def __neg__(self) -> Array:
        from deepmd._vendors import ndtensorflow as xp

        return xp.negative(self)

    def __pos__(self) -> Array:
        from deepmd._vendors import ndtensorflow as xp

        return xp.positive(self)


def _is_supported_operand(value: Any) -> bool:
    return isinstance(value, Array | tf.Tensor | bool | int | float | complex)


def _normalize_index_key(key: Any) -> Any:
    if isinstance(key, Array):
        tensor = key.unwrap()
        if tensor.dtype.is_integer and tensor.shape.rank == 0:
            return tf.cast(tensor, tf.int64)
        return tensor
    if isinstance(key, tuple):
        return tuple(_normalize_index_key(item) for item in key)
    if isinstance(key, slice):
        return slice(
            _normalize_slice_bound(key.start),
            _normalize_slice_bound(key.stop),
            _normalize_slice_bound(key.step),
        )
    return key


def _normalize_slice_bound(value: Any) -> Any:
    value = _normalize_index_key(value)
    if isinstance(value, tf.Tensor) and value.dtype.is_integer:
        return tf.cast(value, tf.int64)
    return value


def _is_integer_index_tensor(key: tf.Tensor) -> bool:
    return key.dtype in (tf.int32, tf.int64)


def _normalize_integer_index(index: tf.Tensor, dim: int | None) -> tf.Tensor:
    index = tf.cast(index, tf.int64)
    if dim is None:
        return index
    return tf.where(index < 0, index + tf.cast(dim, tf.int64), index)


def _advanced_integer_getitem(tensor: tf.Tensor, key: Any) -> tf.Tensor | None:
    if not isinstance(key, tuple):
        return None
    if not any(
        isinstance(item, tf.Tensor) and _is_integer_index_tensor(item) for item in key
    ):
        return None
    if all(not isinstance(item, tf.Tensor) or item.shape.rank == 0 for item in key):
        return None
    if any(
        not (
            isinstance(item, int)
            or isinstance(item, tf.Tensor)
            and _is_integer_index_tensor(item)
        )
        for item in key
    ):
        return None
    if len(key) > tensor.shape.rank:
        return None

    broadcast_shape = tf.TensorShape(())
    for item in key:
        if isinstance(item, tf.Tensor):
            broadcast_shape = tf.broadcast_static_shape(broadcast_shape, item.shape)
    out_shape = tuple(broadcast_shape.as_list())

    coords = []
    for axis, item in enumerate(key):
        dim = tensor.shape[axis]
        if isinstance(item, int):
            item = item + dim if item < 0 and dim is not None else item
            coord = tf.fill(out_shape, tf.cast(item, tf.int64))
        else:
            coord = tf.broadcast_to(_normalize_integer_index(item, dim), out_shape)
        coords.append(coord)
    indices = tf.stack(coords, axis=-1)
    return tf.gather_nd(tensor, indices)


def _scalar_integer_getitem(tensor: tf.Tensor, key: Any) -> tf.Tensor | None:
    if key == ():
        return tensor if tensor.shape.rank == 0 else None
    if not isinstance(key, tuple):
        return None
    if len(key) != tensor.shape.rank:
        return None
    if tensor.shape.rank <= 7:
        return None
    if not all(isinstance(item, int) for item in key):
        return None
    coords = []
    for axis, item in enumerate(key):
        dim = tensor.shape[axis]
        item = item + dim if item < 0 and dim is not None else item
        coords.append(item)
    out = tensor
    for coord in coords:
        out = tf.gather(out, coord, axis=0)
    return out


def _value_to_tensor(value: Any, dtype: tf.DType) -> tf.Tensor:
    if isinstance(value, Array):
        return tf.cast(value.unwrap(), dtype)
    return tf.convert_to_tensor(value, dtype=dtype)


def _binary_forward(name: str) -> Callable[[Array, Any], Array]:
    def method(self: Array, other: Any, /) -> Array:
        if not _is_supported_operand(other):
            return NotImplemented
        from deepmd._vendors import ndtensorflow as xp

        return getattr(xp, name)(self, other)

    return method


def _binary_reflected(name: str) -> Callable[[Array, Any], Array]:
    def method(self: Array, other: Any, /) -> Array:
        if not _is_supported_operand(other):
            return NotImplemented
        from deepmd._vendors import ndtensorflow as xp

        return getattr(xp, name)(other, self)

    return method


Array.__add__ = _binary_forward("add")  # type: ignore[attr-defined]
Array.__radd__ = _binary_reflected("add")  # type: ignore[attr-defined]
Array.__and__ = _binary_forward("bitwise_and")  # type: ignore[attr-defined]
Array.__rand__ = _binary_reflected("bitwise_and")  # type: ignore[attr-defined]
Array.__floordiv__ = _binary_forward("floor_divide")  # type: ignore[attr-defined]
Array.__rfloordiv__ = _binary_reflected("floor_divide")  # type: ignore[attr-defined]
Array.__ge__ = _binary_forward("greater_equal")  # type: ignore[attr-defined]
Array.__le__ = _binary_reflected("greater_equal")  # type: ignore[attr-defined]
Array.__gt__ = _binary_forward("greater")  # type: ignore[attr-defined]
Array.__lt__ = _binary_reflected("greater")  # type: ignore[attr-defined]
Array.__lshift__ = _binary_forward("bitwise_left_shift")  # type: ignore[attr-defined]
Array.__rlshift__ = _binary_reflected("bitwise_left_shift")  # type: ignore[attr-defined]
Array.__matmul__ = _binary_forward("matmul")  # type: ignore[attr-defined]
Array.__rmatmul__ = _binary_reflected("matmul")  # type: ignore[attr-defined]
Array.__mod__ = _binary_forward("remainder")  # type: ignore[attr-defined]
Array.__rmod__ = _binary_reflected("remainder")  # type: ignore[attr-defined]
Array.__mul__ = _binary_forward("multiply")  # type: ignore[attr-defined]
Array.__rmul__ = _binary_reflected("multiply")  # type: ignore[attr-defined]
Array.__or__ = _binary_forward("bitwise_or")  # type: ignore[attr-defined]
Array.__ror__ = _binary_reflected("bitwise_or")  # type: ignore[attr-defined]
Array.__pow__ = _binary_forward("pow")  # type: ignore[attr-defined]
Array.__rpow__ = _binary_reflected("pow")  # type: ignore[attr-defined]
Array.__rshift__ = _binary_forward("bitwise_right_shift")  # type: ignore[attr-defined]
Array.__rrshift__ = _binary_reflected("bitwise_right_shift")  # type: ignore[attr-defined]
Array.__sub__ = _binary_forward("subtract")  # type: ignore[attr-defined]
Array.__rsub__ = _binary_reflected("subtract")  # type: ignore[attr-defined]
Array.__truediv__ = _binary_forward("divide")  # type: ignore[attr-defined]
Array.__rtruediv__ = _binary_reflected("divide")  # type: ignore[attr-defined]
Array.__xor__ = _binary_forward("bitwise_xor")  # type: ignore[attr-defined]
Array.__rxor__ = _binary_reflected("bitwise_xor")  # type: ignore[attr-defined]


__all__ = ["Array", "DLDeviceType"]
