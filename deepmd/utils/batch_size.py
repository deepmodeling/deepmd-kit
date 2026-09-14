# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Callable,
)
from typing import (
    Any,
    Protocol,
)

import array_api_compat

from deepmd.utils.errors import (
    OutOfMemoryError,
)

log = logging.getLogger(__name__)


class RetrySignal(Exception):
    """Signal to retry execution after OOM error."""


class BatchSizeBudget(Protocol):
    """Limit atom budgets and observe completed batch attempts."""

    def limit(self, batch_size: int) -> int:
        """Return an admissible atom budget no larger than the proposal."""
        ...

    def observe(self, nframes: int) -> None:
        """Record an attempt; zero frames denotes an unsuccessful call."""
        ...


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
    silent : bool, default: False
        whether to suppress auto batch size informational logs

    Attributes
    ----------
    current_batch_size : int
        current batch size (number of total atoms)
    maximum_working_batch_size : int
        maximum working batch size
    minimal_not_working_batch_size : int
        minimal not working batch size
    """

    def __init__(
        self,
        initial_batch_size: int = 1024,
        factor: float = 2.0,
        *,
        silent: bool = False,
    ) -> None:
        # See also PyTorchLightning/pytorch-lightning#1638
        self.silent = silent
        self.current_batch_size = initial_batch_size
        DP_INFER_BATCH_SIZE = int(os.environ.get("DP_INFER_BATCH_SIZE", 0))
        self._fixed_batch_size = DP_INFER_BATCH_SIZE > 0
        self._budget: BatchSizeBudget | None = None
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
                if not self.silent:
                    log.warning(
                        "You can use the environment variable DP_INFER_BATCH_SIZE to"
                        "control the inference batch size (nframes * natoms). "
                        f"The default value is {initial_batch_size}."
                    )

        self.factor = factor
        self.oom_retry_mode = False

    def execute(
        self,
        callable: Callable[[int, int], tuple[int, Any]],
        start_index: int,
        natoms: int,
        *,
        max_nframes: int | None = None,
    ) -> tuple[int, Any]:
        """Execute one batch within the search and backend resource budgets.

        Parameters
        ----------
        callable : Callable
            The method should accept the batch size and start_index as parameters,
            and returns executed batch size and data.
        start_index : int
            start index
        natoms : int
            Number of atoms per frame; non-positive values use frame units.
        max_nframes : int or None, optional
            Number of frames remaining in the input, when known.

        Returns
        -------
        int
            Number of frames evaluated, or zero after a recoverable OOM.
        Any
            Result from callable, or None after a recoverable OOM.

        Raises
        ------
        OutOfMemoryError
            OOM when batch size is 1
        """
        if max_nframes is not None and max_nframes <= 0:
            raise ValueError("max_nframes must be positive")
        units_per_frame = max(natoms, 1)
        budget = self._get_batch_budget(natoms, max_nframes)
        if budget is not None:
            if budget is not self._budget:
                self._budget = budget
                self.maximum_working_batch_size = 0
                self.minimal_not_working_batch_size = 2**31
            self._set_batch_size(budget.limit(self.current_batch_size))
        batch_nframes = max(self.current_batch_size // units_per_frame, 1)
        if max_nframes is not None:
            batch_nframes = min(batch_nframes, max_nframes)
        n_batch = 0
        try:
            n_batch, result = callable(batch_nframes, start_index)
        except Exception as e:
            if not self.is_oom_error(e):
                raise
            attempted_size = batch_nframes * units_per_frame
            self.minimal_not_working_batch_size = min(
                self.minimal_not_working_batch_size, attempted_size
            )
            if self.maximum_working_batch_size >= self.minimal_not_working_batch_size:
                self.maximum_working_batch_size = int(
                    self.minimal_not_working_batch_size / self.factor
                )
            if batch_nframes == 1:
                raise OutOfMemoryError(
                    "The callable still throws an out-of-memory (OOM) error even when batch size is 1!"
                ) from e
            self._set_batch_size(
                max(units_per_frame, int(attempted_size / self.factor))
            )
            if self.oom_retry_mode:
                raise RetrySignal from e
            return 0, None
        finally:
            if budget is not None:
                budget.observe(n_batch)

        n_tot = n_batch * units_per_frame
        self.maximum_working_batch_size = max(self.maximum_working_batch_size, n_tot)
        if (
            n_tot + units_per_frame > self.current_batch_size
            and self.current_batch_size * self.factor
            < self.minimal_not_working_batch_size
        ):
            proposed_size = int(self.current_batch_size * self.factor)
            self._set_batch_size(
                proposed_size if budget is None else budget.limit(proposed_size)
            )
        return n_batch, result

    def _get_batch_budget(
        self, natoms: int, max_nframes: int | None
    ) -> BatchSizeBudget | None:
        """Return a backend resource budget, or None for OOM-driven sizing."""
        return None

    def _set_batch_size(self, batch_size: int) -> None:
        old_batch_size = self.current_batch_size
        self.current_batch_size = batch_size
        if not self.silent and self.current_batch_size != old_batch_size:
            log.info(
                f"Adjust batch size from {old_batch_size} to {self.current_batch_size}"
            )

    def execute_all(
        self,
        callable: Callable,
        total_size: int,
        natoms: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a method with all given data.

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
        ) -> tuple[int, Any]:
            end_index = start_index + batch_size
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
            n_batch, result = self.execute(
                execute_with_batch_size,
                index,
                natoms,
                max_nframes=total_size - index,
            )
            if n_batch == 0:
                continue
            returned_dict = (
                isinstance(result, dict) if returned_dict is None else returned_dict
            )
            if not returned_dict:
                result = (result,) if not isinstance(result, tuple) else result
            index += n_batch

            def append_to_list(res_list: list[Any], res: Any) -> list[Any]:
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

        def concate_result(r: list[Any]) -> Any:
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
            r_list = [concate_result(r) for r in zip(*results, strict=True)]
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

    def set_oom_retry_mode(self, enable: bool) -> None:
        """Set OOM retry mode.

        In OOM retry mode, an OOM during execution may reduce the current
        batch size and raise :class:`RetrySignal` to indicate that execution
        should be retried.

        Callers that want all data to be re-executed must catch
        :class:`RetrySignal` and restart the full evaluation themselves.

        Parameters
        ----------
        enable : bool
            True to enable OOM retry mode
        """
        self.oom_retry_mode = enable
