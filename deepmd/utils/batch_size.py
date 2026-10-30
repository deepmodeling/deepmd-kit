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
)

import array_api_compat

from deepmd.utils.errors import (
    OutOfMemoryError,
)

log = logging.getLogger(__name__)


class RetrySignal(Exception):
    """Signal to retry execution after OOM error."""


class AutoBatchSize(ABC):
    """Grow inference batches and reuse the learned out-of-memory bound.

    Successful full batches grow geometrically. After OOM, growth beyond the
    failed bound resumes after two successful full batches; repeated recovery probes
    extend that interval to four, then eight batches. This deliberately favors
    recovering capacity over permanently avoiding OOM. Short tails and changes
    in the number of atoms do not reset the learned state. No allocator profiling
    or exact maximum search is needed.

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
        self._auto_growth = DP_INFER_BATCH_SIZE <= 0 and self.is_gpu_available()
        self._successful_batches = 0
        self._reprobe_interval = 2
        if DP_INFER_BATCH_SIZE > 0:
            self.current_batch_size = DP_INFER_BATCH_SIZE
            self.maximum_working_batch_size = DP_INFER_BATCH_SIZE
            self.minimal_not_working_batch_size = self.maximum_working_batch_size + 1
        else:
            self.maximum_working_batch_size = initial_batch_size
            if self._auto_growth:
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
        """Execute one batch, growing on success and backing off on OOM.

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
        batch_nframes = max(self.current_batch_size // units_per_frame, 1)
        if max_nframes is not None:
            batch_nframes = min(batch_nframes, max_nframes)
        try:
            n_batch, result = callable(batch_nframes, start_index)
        except Exception as e:
            if not self.is_oom_error(e):
                raise
            attempted_size = batch_nframes * units_per_frame
            self._successful_batches = 0
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

        n_tot = n_batch * units_per_frame
        self.maximum_working_batch_size = max(self.maximum_working_batch_size, n_tot)
        # A successful *actual* batch can disprove an old OOM bound. A
        # truncated probe cannot: the proposed size was not actually tested.
        if self._auto_growth and n_tot >= self.minimal_not_working_batch_size:
            self.minimal_not_working_batch_size = 2**31
        if self._auto_growth and n_tot + units_per_frame > self.current_batch_size:
            self._successful_batches += 1
            self._grow_batch_size(n_tot, allow_reprobe=not self.oom_retry_mode)
        return n_batch, result

    def _grow_batch_size(self, successful_size: int, *, allow_reprobe: bool) -> None:
        """Grow below the failed bound, or schedule an allowed recovery probe."""
        # Large mandatory single frames also justify immediate growth.
        next_batch_size = int(
            max(self.current_batch_size, successful_size) * self.factor
        )
        if next_batch_size >= self.minimal_not_working_batch_size:
            if not allow_reprobe or self._successful_batches < self._reprobe_interval:
                return
            # Dampen repeated recovery probes, including oscillating workloads.
            self._reprobe_interval = min(2 * self._reprobe_interval, 8)
        self._set_batch_size(next_batch_size)
        self._successful_batches = 0

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
        # Retry-mode callers discard all hook outputs on OOM and restart this
        # entire input. Probing a known failure mid-pass could restart forever.
        # Defer that probe until the pass is complete, for the next invocation.
        if (
            self._auto_growth
            and self.oom_retry_mode
            and self._successful_batches >= self._reprobe_interval
        ):
            self._grow_batch_size(0, allow_reprobe=True)
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
        Recovery probes beyond a known failed size are deferred until
        :meth:`execute_all` has assembled its complete result, for the next
        invocation. Direct :meth:`execute` calls have no completion boundary
        and therefore do not schedule those probes in retry mode.

        Parameters
        ----------
        enable : bool
            True to enable OOM retry mode
        """
        self.oom_retry_mode = enable
