# SPDX-License-Identifier: LGPL-3.0-or-later


from deepmd.jax.env import (
    jax,
)
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
    silent : bool, default: False
        Whether to suppress batch-size adjustment logs.
    maximum_batch_size : int or None, optional
        Upper bound on the total atom budget, limiting the static shapes
        compiled during repeated evaluation. ``None`` retains unrestricted
        GPU batch growth. An explicit ``DP_INFER_BATCH_SIZE`` takes precedence.
        OOM failures still reduce the batch size. Frames are indivisible, so
        one frame may contain more atoms than the budget.

    Raises
    ------
    ValueError
        If ``maximum_batch_size`` is non-positive.

    """

    def __init__(
        self,
        initial_batch_size: int = 1024,
        factor: float = 2.0,
        *,
        silent: bool = False,
        maximum_batch_size: int | None = None,
    ) -> None:
        if maximum_batch_size is not None:
            if maximum_batch_size <= 0:
                raise ValueError(
                    f"maximum_batch_size must be positive; got {maximum_batch_size}."
                )
            initial_batch_size = min(initial_batch_size, maximum_batch_size)
        super().__init__(
            initial_batch_size=initial_batch_size,
            factor=factor,
            silent=silent,
        )
        # Only a growing budget can pass the ceiling: an explicit
        # DP_INFER_BATCH_SIZE fixes the size, and without growth the budget
        # never exceeds the capped initial size.
        self.maximum_batch_size = maximum_batch_size if self._auto_growth else None

    def _set_batch_size(self, batch_size: int) -> None:
        if self.maximum_batch_size is not None:
            batch_size = min(batch_size, self.maximum_batch_size)
        super()._set_batch_size(batch_size)

    def is_gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns
        -------
        bool
            True if GPU is available
        """
        return jax.devices()[0].platform == "gpu"

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
        if isinstance(e, (RuntimeError, ValueError)) and (
            "RESOURCE_EXHAUSTED:" in e.args[0]
        ):
            return True
        return False
