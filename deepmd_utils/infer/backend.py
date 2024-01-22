# SPDX-License-Identifier: LGPL-3.0-or-later
from enum import (
    Enum,
)


class DPBackend(Enum):
    """DeePMD-kit backend."""

    TensorFlow = 1
    PyTorch = 2
    Paddle = 3
    Unknown = 4


def detect_backend(filename: str) -> DPBackend:
    """Detect the backend of the given model file.

    Parameters
    ----------
    filename : str
        The model file name
    """
    if filename.endswith(".pb"):
        return DPBackend.TensorFlow
    elif filename.endswith(".pth") or filename.endswith(".pt"):
        return DPBackend.PyTorch
    elif filename.endswith(".pdmodel"):
        return DPBackend.Paddle
    return DPBackend.Unknown


__all__ = ["DPBackend", "detect_backend"]
