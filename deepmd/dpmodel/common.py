# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)

import numpy as np

from deepmd.common import (
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
    "default": GLOBAL_NP_FLOAT_PRECISION,
}
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
