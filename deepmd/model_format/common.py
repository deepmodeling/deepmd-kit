# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
)

import numpy as np

PRECISION_DICT = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "half": np.float16,
    "single": np.float32,
    "double": np.float64,
}
DEFAULT_PRECISION = "float64"


class NativeOP(ABC):
    """The unit operation of a native model."""

    def call(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        return self.call(*args, **kwargs)
