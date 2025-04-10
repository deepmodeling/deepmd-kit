# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.utils.batch_size import AutoBatchSize as AutoBatchSizeBase


class AutoBatchSize(AutoBatchSizeBase):
    """Auto batch size.

    Parameters
    ----------
    initial_batch_size : int, default: 1024
        initial batch size (number of total atoms) when DP_INFER_BATCH_SIZE
        is not set
    factor : float, default: 2.
        increased factor

    """

    def __init__(
        self,
        initial_batch_size: int = 1024,
        factor: float = 2.0,
    ) -> None:
        super().__init__(
            initial_batch_size=initial_batch_size,
            factor=factor,
        )

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
        # several sources think CUSOLVER_STATUS_INTERNAL_ERROR is another out-of-memory error,
        # such as https://github.com/JuliaGPU/CUDA.jl/issues/1924
        # (the meaningless error message should be considered as a bug in cusolver)
        if (
            isinstance(e, RuntimeError)
            and (
                "CUDA out of memory." in e.args[0]
                or "CUDA driver error: out of memory" in e.args[0]
                or "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR" in e.args[0]
                # https://github.com/deepmodeling/deepmd-kit/issues/4594
                or "CUDA error: out of memory" in e.args[0]
            )
        ) or isinstance(e, torch.cuda.OutOfMemoryError):
            # Release all unoccupied cached memory
            torch.cuda.empty_cache()
            return True
        return False
