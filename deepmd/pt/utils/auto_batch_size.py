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
        *,
        silent: bool = False,
    ) -> None:
        super().__init__(
            initial_batch_size=initial_batch_size,
            factor=factor,
            silent=silent,
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
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return True

        if not isinstance(e, RuntimeError) or not e.args:
            return False

        # Gather messages from the exception itself and its chain.  AOTInductor
        # (.pt2) sometimes strips the underlying OOM message when rewrapping,
        # but not always; checking ``__cause__`` / ``__context__`` catches the
        # remaining cases when the original error is preserved.
        msgs: list[str] = []
        cur: BaseException | None = e
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if cur.args:
                first = cur.args[0]
                if isinstance(first, str):
                    msgs.append(first)
            cur = cur.__cause__ or cur.__context__

        # Several sources treat CUSOLVER_STATUS_INTERNAL_ERROR as an OOM, e.g.
        # https://github.com/JuliaGPU/CUDA.jl/issues/1924
        plain_oom_markers = (
            "CUDA out of memory.",
            "CUDA driver error: out of memory",
            "CUDA error: out of memory",
            "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR",
        )
        if any(m in msg for msg in msgs for m in plain_oom_markers):
            torch.cuda.empty_cache()
            return True

        # AOTInductor (.pt2) wraps the underlying CUDA OOM as a generic
        # ``run_func_(...) API call failed at .../model_container_runner.cpp``.
        # The original "CUDA out of memory" text is printed to stderr only and
        # is absent from the Python-level RuntimeError, so we match on the
        # wrapper signature.  If the root cause turns out to be something
        # other than OOM, ``execute()`` will keep shrinking the batch and
        # eventually raise ``OutOfMemoryError`` at batch size 1, which is a
        # clean failure rather than an uncaught exception.
        aoti_wrapped = any(
            "run_func_(" in msg and "model_container_runner" in msg for msg in msgs
        )
        if aoti_wrapped:
            torch.cuda.empty_cache()
            return True

        return False
