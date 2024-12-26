# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from functools import (
    wraps,
)
from typing import (
    Any,
    Callable,
    Optional,
    overload,
)

import array_api_compat
import ml_dtypes
import numpy as np

from deepmd.common import (
    VALID_PRECISION,
)
from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
)

PRECISION_DICT = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "half": np.float16,
    "single": np.float32,
    "double": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
    "default": GLOBAL_NP_FLOAT_PRECISION,
    # NumPy doesn't have bfloat16 (and doesn't plan to add)
    # ml_dtypes is a solution, but it seems not supporting np.save/np.load
    # hdf5 hasn't supported bfloat16 as well (see https://forum.hdfgroup.org/t/11975)
    "bfloat16": ml_dtypes.bfloat16,
}
assert VALID_PRECISION.issubset(PRECISION_DICT.keys())

RESERVED_PRECISION_DICT = {
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.int32: "int32",
    np.int64: "int64",
    ml_dtypes.bfloat16: "bfloat16",
    np.bool_: "bool",
}
assert set(RESERVED_PRECISION_DICT.keys()) == set(PRECISION_DICT.values())
DEFAULT_PRECISION = "float64"


def get_xp_precision(
    xp: Any,
    precision: str,
):
    """Get the precision from the API compatible namespace."""
    if precision == "float16" or precision == "half":
        return xp.float16
    elif precision == "float32" or precision == "single":
        return xp.float32
    elif precision == "float64" or precision == "double":
        return xp.float64
    elif precision == "int32":
        return xp.int32
    elif precision == "int64":
        return xp.int64
    elif precision == "bool":
        return bool
    elif precision == "default":
        return get_xp_precision(xp, RESERVED_PRECISION_DICT[PRECISION_DICT[precision]])
    elif precision == "global":
        return get_xp_precision(xp, RESERVED_PRECISION_DICT[GLOBAL_NP_FLOAT_PRECISION])
    elif precision == "bfloat16":
        return ml_dtypes.bfloat16
    else:
        raise ValueError(f"unsupported precision {precision} for {xp}")


class NativeOP(ABC):
    """The unit operation of a native model."""

    @abstractmethod
    def call(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        pass

    def __call__(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        return self.call(*args, **kwargs)


def to_numpy_array(x: Any) -> Optional[np.ndarray]:
    """Convert an array to a NumPy array.

    Parameters
    ----------
    x : Any
        The array to be converted.

    Returns
    -------
    Optional[np.ndarray]
        The NumPy array.
    """
    if x is None:
        return None
    try:
        # asarray is not within Array API standard, so may fail
        return np.asarray(x)
    except (ValueError, AttributeError):
        xp = array_api_compat.array_namespace(x)
        # to fix BufferError: Cannot export readonly array since signalling readonly is unsupported by DLPack.
        x = xp.asarray(x, copy=True)
        return np.from_dlpack(x)


def cast_precision(func: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that casts and casts back the input
    and output tensor of a method.

    The decorator should be used on an instance method.

    The decorator will do the following thing:
    (1) It casts input arrays from the global precision
    to precision defined by property `precision`.
    (2) It casts output arrays from `precision` to
    the global precision.
    (3) It checks inputs and outputs and only casts when
    input or output is an array and its dtype matches
    the global precision and `precision`, respectively.
    If it does not match (e.g. it is an integer), the decorator
    will do nothing on it.

    The decorator supports the array API.

    Returns
    -------
    Callable
        a decorator that casts and casts back the input and
        output array of a method

    Examples
    --------
    >>> class A:
    ...     def __init__(self):
    ...         self.precision = "float32"
    ...
    ...     @cast_precision
    ...     def f(x: Array, y: Array) -> Array:
    ...         return x**2 + y
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # only convert tensors
        returned_tensor = func(
            self,
            *[safe_cast_array(vv, "global", self.precision) for vv in args],
            **{
                kk: safe_cast_array(vv, "global", self.precision)
                for kk, vv in kwargs.items()
            },
        )
        if isinstance(returned_tensor, tuple):
            return tuple(
                safe_cast_array(vv, self.precision, "global") for vv in returned_tensor
            )
        elif isinstance(returned_tensor, dict):
            return {
                kk: safe_cast_array(vv, self.precision, "global")
                for kk, vv in returned_tensor.items()
            }
        else:
            return safe_cast_array(returned_tensor, self.precision, "global")

    return wrapper


@overload
def safe_cast_array(
    input: np.ndarray, from_precision: str, to_precision: str
) -> np.ndarray: ...
@overload
def safe_cast_array(input: None, from_precision: str, to_precision: str) -> None: ...
def safe_cast_array(
    input: Optional[np.ndarray], from_precision: str, to_precision: str
) -> Optional[np.ndarray]:
    """Convert an array from a precision to another precision.

    If input is not an array or without the specific precision, the method will not
    cast it.

    Array API is supported.

    Parameters
    ----------
    input : np.ndarray or None
        Input array
    from_precision : str
        Array data type that is casted from
    to_precision : str
        Array data type that casts to

    Returns
    -------
    np.ndarray or None
        casted array
    """
    if array_api_compat.is_array_api_obj(input):
        xp = array_api_compat.array_namespace(input)
        if input.dtype == get_xp_precision(xp, from_precision):
            return xp.astype(input, get_xp_precision(xp, to_precision))
    return input


__all__ = [
    "DEFAULT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "PRECISION_DICT",
    "RESERVED_PRECISION_DICT",
    "NativeOP",
]
