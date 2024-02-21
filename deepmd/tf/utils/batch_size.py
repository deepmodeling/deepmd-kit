# SPDX-License-Identifier: LGPL-3.0-or-later
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


class AutoBatchSize(AutoBatchSizeBase):
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
        # TODO: it's very slow to catch OOM error; I don't know what TF is doing here
        # but luckily we only need to catch once
        return isinstance(e, (tf.errors.ResourceExhaustedError, OutOfMemoryError))
