# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Tuple,
    Union,
)

import numpy as np
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
        # several sources think CUSOLVER_STATUS_INTERNAL_ERROR is another out-of-memory error,
        # such as https://github.com/JuliaGPU/CUDA.jl/issues/1924
        # (the meaningless error message should be considered as a bug in cusolver)
        if isinstance(e, RuntimeError) and (
            "CUDA out of memory." in e.args[0]
            or "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR" in e.args[0]
        ):
            # Release all unoccupied cached memory
            torch.cuda.empty_cache()
            return True
        return False

    def execute_all(
        self, callable: Callable, total_size: int, natoms: int, *args, **kwargs
    ) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        """Excuate a method with all given data.

        Parameters
        ----------
        callable : Callable
            The method should accept *args and **kwargs as input and return the similiar array.
        total_size : int
            Total size
        natoms : int
            The number of atoms
        *args
            Variable length argument list.
        **kwargs
            If 2D np.ndarray or torch.Tensor, assume the first axis is batch; otherwise do nothing.
        """

        def execute_with_batch_size(
            batch_size: int, start_index: int
        ) -> Tuple[int, Tuple[torch.Tensor]]:
            end_index = start_index + batch_size
            end_index = min(end_index, total_size)
            return (end_index - start_index), callable(
                *[
                    (
                        vv[start_index:end_index]
                        if (isinstance(vv, np.ndarray) or isinstance(vv, torch.Tensor))
                        and vv.ndim > 1
                        else vv
                    )
                    for vv in args
                ],
                **{
                    kk: (
                        vv[start_index:end_index]
                        if (isinstance(vv, np.ndarray) or isinstance(vv, torch.Tensor))
                        and vv.ndim > 1
                        else vv
                    )
                    for kk, vv in kwargs.items()
                },
            )

        index = 0
        results = []
        while index < total_size:
            n_batch, result = self.execute(execute_with_batch_size, index, natoms)
            if not isinstance(result, tuple):
                result = (result,)
            index += n_batch
            if n_batch:
                for rr in result:
                    rr.reshape((n_batch, -1))
                results.append(result)
        r_list = []
        for r in zip(*results):
            if isinstance(r[0], np.ndarray):
                r_list.append(np.concatenate(r, axis=0))
            elif isinstance(r[0], torch.Tensor):
                r_list.append(torch.cat(r, dim=0))
            else:
                raise RuntimeError(f"Unexpected result type {type(r[0])}")
        r = tuple(r_list)
        if len(r) == 1:
            # avoid returning tuple if callable doesn't return tuple
            r = r[0]
        return r
