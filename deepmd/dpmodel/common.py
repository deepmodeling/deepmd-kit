# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Optional,
)

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
    "bool": bool,
    "default": GLOBAL_NP_FLOAT_PRECISION,
    # NumPy doesn't have bfloat16 (and doesn't plan to add)
    # ml_dtypes is a solution, but it seems not supporting np.save/np.load
    # hdf5 hasn't supported bfloat16 as well (see https://forum.hdfgroup.org/t/11975)
    "bfloat16": ml_dtypes.bfloat16,
}
assert VALID_PRECISION.issubset(PRECISION_DICT.keys())

RESERVED_PRECISON_DICT = {
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.int32: "int32",
    np.int64: "int64",
    ml_dtypes.bfloat16: "bfloat16",
    bool: "bool",
}
assert set(RESERVED_PRECISON_DICT.keys()) == set(PRECISION_DICT.values())
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
        return get_xp_precision(xp, RESERVED_PRECISON_DICT[PRECISION_DICT[precision]])
    elif precision == "global":
        return get_xp_precision(xp, RESERVED_PRECISON_DICT[GLOBAL_NP_FLOAT_PRECISION])
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
    if hasattr(x, "__dlpack_device__") and x.__dlpack_device__()[0] == 1:
        # CPU = 1, see https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html#api-specification-generated-array-api-array-dlpack-device--page-root
        # dlpack needs the device to be the same
        return np.from_dlpack(x)
    # asarray is not within Array API standard, so may fail
    return np.asarray(x)


__all__ = [
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "PRECISION_DICT",
    "RESERVED_PRECISON_DICT",
    "DEFAULT_PRECISION",
    "NativeOP",
]
