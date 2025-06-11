# SPDX-License-Identifier: LGPL-3.0-or-later

import paddle

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
        return paddle.device.cuda.device_count() > 0

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
        if isinstance(e, MemoryError) and ("ResourceExhaustedError" in e.args[0]):
            # Release all unoccupied cached memory
            paddle.device.cuda.empty_cache()
            return True
        return False
