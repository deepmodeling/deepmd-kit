# SPDX-License-Identifier: LGPL-3.0-or-later
"""Automatic inference batch sizing for TensorFlow 2."""

from deepmd.tf2.env import (
    tf,
)
from deepmd.utils.batch_size import AutoBatchSize as AutoBatchSizeBase


class AutoBatchSize(AutoBatchSizeBase):
    """Auto batch size helper for TF2 eager inference."""

    def is_gpu_available(self) -> bool:
        """Return whether a GPU is visible to TensorFlow."""
        return bool(tf.config.list_physical_devices("GPU"))

    def is_oom_error(self, e: Exception) -> bool:
        """Return whether an exception is TensorFlow's OOM signal."""
        return isinstance(e, tf.errors.ResourceExhaustedError)
