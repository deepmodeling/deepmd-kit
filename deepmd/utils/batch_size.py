# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Callable,
)

import array_api_compat
import numpy as np

from deepmd.utils.errors import (
    OutOfMemoryError,
)

log = logging.getLogger(__name__)


class AutoBatchSize(ABC):
    """This class allows DeePMD-kit to automatically decide the maximum
    batch size that will not cause an OOM error.

    Notes
    -----
    In some CPU environments, the program may be directly killed when OOM. In
    this case, by default the batch size will not be increased for CPUs. The
    environment variable `DP_INFER_BATCH_SIZE` can be set as the batch size.

    In other cases, we assume all OOM error will raise :class:`OutOfMemoryError`.

    Parameters
    ----------
    initial_batch_size : int, default: 1024
        initial batch size (number of total atoms) when DP_INFER_BATCH_SIZE
        is not set
    factor : float, default: 2.
        increased factor

    Attributes
    ----------
    current_batch_size : int
        current batch size (number of total atoms)
    maximum_working_batch_size : int
        maximum working batch size
    minimal_not_working_batch_size : int
        minimal not working batch size
    """

    def __init__(self, initial_batch_size: int = 1024, factor: float = 2.0) -> None:
        # See also PyTorchLightning/pytorch-lightning#1638
        self.current_batch_size = initial_batch_size
        DP_INFER_BATCH_SIZE = int(os.environ.get("DP_INFER_BATCH_SIZE", 0))
        if DP_INFER_BATCH_SIZE > 0:
            self.current_batch_size = DP_INFER_BATCH_SIZE
            self.maximum_working_batch_size = DP_INFER_BATCH_SIZE
            self.minimal_not_working_batch_size = self.maximum_working_batch_size + 1
        else:
            self.maximum_working_batch_size = initial_batch_size
            if self.is_gpu_available():
                self.minimal_not_working_batch_size = 2**31
            else:
                self.minimal_not_working_batch_size = (
                    self.maximum_working_batch_size + 1
                )
                log.warning(
                    "You can use the environment variable DP_INFER_BATCH_SIZE to"
                    "control the inference batch size (nframes * natoms). "
                    f"The default value is {initial_batch_size}."
                )

        self.factor = factor

    def execute(
        self, callable: Callable, start_index: int, natoms: int
    ) -> tuple[int, tuple]:
        """Excuate a method with given batch size.

        Parameters
        ----------
        callable : Callable
            The method should accept the batch size and start_index as parameters,
            and returns executed batch size and data.
        start_index : int
            start index
        natoms : int
            natoms

        Returns
        -------
        int
            executed batch size * number of atoms
        tuple
            result from callable, None if failing to execute

        Raises
        ------
        OutOfMemoryError
            OOM when batch size is 1
        """
        if natoms > 0:
            batch_nframes = self.current_batch_size // natoms
        else:
            batch_nframes = self.current_batch_size
        try:
            n_batch, result = callable(max(batch_nframes, 1), start_index)
        except Exception as e:
            if not self.is_oom_error(e):
                raise e
            self.minimal_not_working_batch_size = min(
                self.minimal_not_working_batch_size, self.current_batch_size
            )
            if self.maximum_working_batch_size >= self.minimal_not_working_batch_size:
                self.maximum_working_batch_size = int(
                    self.minimal_not_working_batch_size / self.factor
                )
            if self.minimal_not_working_batch_size <= natoms:
                raise OutOfMemoryError(
                    "The callable still throws an out-of-memory (OOM) error even when batch size is 1!"
                ) from e
            # adjust the next batch size
            self._adjust_batch_size(1.0 / self.factor)
            return 0, None
        else:
            n_tot = n_batch * natoms
            self.maximum_working_batch_size = max(
                self.maximum_working_batch_size, n_tot
            )
            # adjust the next batch size
            if (
                n_tot + natoms > self.current_batch_size
                and self.current_batch_size * self.factor
                < self.minimal_not_working_batch_size
            ):
                self._adjust_batch_size(self.factor)
            return n_batch, result

    def _adjust_batch_size(self, factor: float) -> None:
        old_batch_size = self.current_batch_size
        self.current_batch_size = int(self.current_batch_size * factor)
        log.info(
            f"Adjust batch size from {old_batch_size} to {self.current_batch_size}"
        )

    def execute_all(
        self, callable: Callable, total_size: int, natoms: int, *args, **kwargs
    ) -> tuple[np.ndarray]:
        """Excuate a method with all given data.

        This method is compatible with Array API.

        Parameters
        ----------
        callable : Callable
            The method should accept *args and **kwargs as input and return the similar array.
        total_size : int
            Total size
        natoms : int
            The number of atoms
        *args
            Variable length argument list.
        **kwargs
            If 2D np.ndarray, assume the first axis is batch; otherwise do nothing.
        """

        def execute_with_batch_size(
            batch_size: int, start_index: int
        ) -> tuple[int, tuple[np.ndarray]]:
            end_index = start_index + batch_size
            end_index = min(end_index, total_size)
            return (end_index - start_index), callable(
                *[
                    (
                        vv[start_index:end_index, ...]
                        if (
                            (array_api_compat.is_array_api_obj(vv) and vv.ndim > 1)
                            or str(vv.__class__) == "<class 'paddle.Tensor'>"
                        )
                        else vv
                    )
                    for vv in args
                ],
                **{
                    kk: (
                        vv[start_index:end_index, ...]
                        if (
                            (array_api_compat.is_array_api_obj(vv) and vv.ndim > 1)
                            or str(vv.__class__) == "<class 'paddle.Tensor'>"
                        )
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
            if n_batch == 0:
                continue
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
                results = {kk: [] for kk in result} if results is None else results
                results = {kk: append_to_list(results[kk], result[kk]) for kk in result}
        assert results is not None
        assert returned_dict is not None

        def concate_result(r):
            if array_api_compat.is_array_api_obj(r[0]):
                xp = array_api_compat.array_namespace(r[0])
                ret = xp.concat(r, axis=0)
            elif str(r[0].__class__) == "<class 'paddle.Tensor'>":
                try:
                    import paddle
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "The 'paddlepaddle' is required but not installed."
                    ) from e
                ret = paddle.concat(r, axis=0)
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

    @abstractmethod
    def is_gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns
        -------
        bool
            True if GPU is available
        """

    @abstractmethod
    def is_oom_error(self, e: Exception) -> bool:
        """Check if the exception is an OOM error.

        Parameters
        ----------
        e : Exception
            Exception

        Returns
        -------
        bool
            True if the exception is an OOM error
        """
