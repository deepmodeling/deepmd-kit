# SPDX-License-Identifier: LGPL-3.0-or-later
import os

from packaging.version import (
    Version,
)

from deepmd.tf.env import (
    TF_VERSION,
    tf,
)
from deepmd.tf.utils.errors import (
    OutOfMemoryError,
)
from deepmd.utils.batch_size import AutoBatchSize as AutoBatchSizeBase
from deepmd.utils.batch_size import (
    log,
)


class AutoBatchSize(AutoBatchSizeBase):
    def __init__(self, initial_batch_size: int = 1024, factor: float = 2.0) -> None:
        super().__init__(initial_batch_size, factor)
        DP_INFER_BATCH_SIZE = int(os.environ.get("DP_INFER_BATCH_SIZE", 0))
        if not DP_INFER_BATCH_SIZE > 0:
            if self.is_gpu_available():
                log.info(
                    "If you encounter the error 'an illegal memory access was encountered', this may be due to a TensorFlow issue. "
                    "To avoid this, set the environment variable DP_INFER_BATCH_SIZE to a smaller value than the last adjusted batch size. "
                    "The environment variable DP_INFER_BATCH_SIZE controls the inference batch size (nframes * natoms). "
                )

    def is_gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns
        -------
        bool
            True if GPU is available
        """
        return (
            Version(TF_VERSION) >= Version("1.14")
            and tf.config.experimental.get_visible_devices("GPU")
        ) or tf.test.is_gpu_available()

    def is_oom_error(self, e: Exception) -> bool:
        """Check if the exception is an OOM error.

        Parameters
        ----------
        e : Exception
            Exception
        """
        return isinstance(e, (tf.errors.ResourceExhaustedError, OutOfMemoryError))
