from abc import ABC
import numpy as np

PRECISION_DICT = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "default": np.float64,
}

class NativeOP(ABC):
    """The unit operation of a native model."""

    def call(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        raise NotImplementedError
