# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from builtins import bool as py_bool
from typing import (
    TypedDict,
)

import tensorflow as tf

from ._namespace import bool as bool_dtype
from ._namespace import (
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    isdtype,
    uint8,
    uint16,
    uint32,
    uint64,
)

DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": tf.DType,
        "complex floating": tf.DType,
        "integral": tf.DType,
        "indexing": tf.DType,
    },
)


class DataTypes(TypedDict, total=False):
    bool: tf.DType
    int8: tf.DType
    int16: tf.DType
    int32: tf.DType
    int64: tf.DType
    uint8: tf.DType
    uint16: tf.DType
    uint32: tf.DType
    uint64: tf.DType
    float32: tf.DType
    float64: tf.DType
    complex64: tf.DType
    complex128: tf.DType


Capabilities = TypedDict(
    "Capabilities",
    {
        "boolean indexing": py_bool,
        "data-dependent shapes": py_bool,
        "max dimensions": int | None,
    },
)


def _device_name(device: tf.config.LogicalDevice) -> str:
    name = device.name
    if name.startswith("/device:"):
        return name.removeprefix("/device:")
    return name


class Info:
    """Namespace returned by ``__array_namespace_info__``."""

    def capabilities(self) -> Capabilities:
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            "max dimensions": None,
        }

    def default_device(self) -> str:
        devices = self.devices()
        return devices[0] if devices else "CPU:0"

    def default_dtypes(self, *, device: str | None = None) -> DefaultDataTypes:
        del device
        return {
            "real floating": float32,
            "complex floating": complex64,
            "integral": int32,
            "indexing": int64,
        }

    def devices(self) -> tuple[str, ...]:
        return tuple(
            _device_name(device) for device in tf.config.list_logical_devices()
        )

    def dtypes(
        self,
        *,
        device: str | None = None,
        kind: None | str | tuple[str, ...] = None,
    ) -> DataTypes:
        del device
        dtypes = {
            "bool": bool_dtype,
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
        if kind is None:
            return dtypes
        return {name: dtype for name, dtype in dtypes.items() if isdtype(dtype, kind)}


def __array_namespace_info__() -> Info:
    return Info()


__all__ = ["Info", "__array_namespace_info__"]
