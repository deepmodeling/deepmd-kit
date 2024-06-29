# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
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
    # NumPy doesn't have bfloat16 (and does't plan to add)
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


class NativeOP(ABC):
    """The unit operation of a native model."""

    @abstractmethod
    def call(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        pass

    def __call__(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        return self.call(*args, **kwargs)


__all__ = [
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "PRECISION_DICT",
    "RESERVED_PRECISON_DICT",
    "DEFAULT_PRECISION",
    "NativeOP",
]
