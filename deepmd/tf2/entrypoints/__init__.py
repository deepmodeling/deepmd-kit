# SPDX-License-Identifier: LGPL-3.0-or-later
"""Entry points for the TensorFlow 2 backend."""

from deepmd.tf2.entrypoints.compress import (
    enable_compression,
)
from deepmd.tf2.entrypoints.freeze import (
    freeze,
)
from deepmd.tf2.entrypoints.train import (
    train,
)

__all__ = ["enable_compression", "freeze", "train"]
