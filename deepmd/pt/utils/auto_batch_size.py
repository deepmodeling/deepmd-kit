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
    ):
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
        if isinstance(e, RuntimeError) and (
            "CUDA out of memory." in e.args[0]
            or "CUDA driver error: out of memory" in e.args[0]
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
        results = None
        returned_dict = None
        while index < total_size:
            n_batch, result = self.execute(execute_with_batch_size, index, natoms)
            returned_dict = (
                isinstance(result, dict) if returned_dict is None else returned_dict
            )
            if not returned_dict:
                result = (result,) if not isinstance(result, tuple) else result
            index += n_batch

            def append_to_list(res_list, res):
                if n_batch:
                    res_list.append(res)
                return res_list

            if not returned_dict:
                results = [] if results is None else results
                results = append_to_list(results, result)
            else:
                results = (
                    {kk: [] for kk in result.keys()} if results is None else results
                )
                results = {
                    kk: append_to_list(results[kk], result[kk]) for kk in result.keys()
                }
        assert results is not None
        assert returned_dict is not None

        def concate_result(r):
            if isinstance(r[0], np.ndarray):
                ret = np.concatenate(r, axis=0)
            elif isinstance(r[0], torch.Tensor):
                ret = torch.cat(r, dim=0)
            else:
                raise RuntimeError(f"Unexpected result type {type(r[0])}")
            return ret

        if not returned_dict:
            r_list = [concate_result(r) for r in zip(*results)]
            r = tuple(r_list)
            if len(r) == 1:
                # avoid returning tuple if callable doesn't return tuple
                r = r[0]
        else:
            r = {kk: concate_result(vv) for kk, vv in results.items()}
        return r
