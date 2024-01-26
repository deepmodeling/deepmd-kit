# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.utils.batch_size import AutoBatchSize as AutoBatchSizeBase


class AutoBatchSize(AutoBatchSizeBase):
    def is_gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns
        -------
        bool
            True if GPU is available
        """
        return torch.cuda.is_available()

    def is_oom_error(self, e: Exception) -> bool:
        """Check if the exception is an OOM error.

        Parameters
        ----------
        e : Exception
            Exception
        """
        return isinstance(e, RuntimeError) and "CUDA out of memory." in e.args[0]
