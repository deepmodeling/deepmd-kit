# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import (
    Any,
)

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.utils.batch_size import AutoBatchSize as AutoBatchSizeBase
from deepmd.utils.batch_size import (
    BatchSizeBudget,
)


class _CudaMemoryBudget:
    """Estimate batch workspace without modifying allocator peak statistics.

    Calibration releases unused cache once. Reserved-memory growth together
    with the remaining default-pool cache bounds batch workspace. Private CUDA
    pools remain reserved for their graphs and are not reusable by evaluation.
    """

    def __init__(self, device: int, natoms: int, stats: dict[str, Any]) -> None:
        self.device = device
        self.natoms = natoms
        self.reserved = stats["reserved_bytes"]["all"]["current"]
        self.allocated = stats["allocated_bytes"]["all"]["current"]
        self.allocated_total = stats["allocated_bytes"]["all"]["allocated"]
        self.peak = stats["allocated_bytes"]["all"]["peak"]
        self.cached = self._cached_bytes()
        self.measured_atoms = 0
        self.bytes_per_atom = 0.0
        self.device_events = self._device_events(stats)

    @staticmethod
    def _device_events(stats: dict[str, Any]) -> tuple[int, int]:
        return stats["num_device_alloc"], stats["num_device_free"]

    def matches(self, device: int, natoms: int, stats: dict[str, Any]) -> bool:
        return (
            self.device == device
            and self.natoms == natoms
            and self.device_events == self._device_events(stats)
        )

    def observe(self, nframes: int) -> None:
        stats = torch.cuda.memory_stats_as_nested_dict(self.device)
        self.device_events = self._device_events(stats)
        allocated_total = stats["allocated_bytes"]["all"]["allocated"]
        allocated = allocated_total - self.allocated_total
        self.allocated_total = allocated_total
        atoms = nframes * self.natoms
        if atoms == 0 or atoms < self.measured_atoms:
            return
        workspace = max(
            0, stats["reserved_bytes"]["all"]["current"] - self.reserved + self.cached
        )
        peak = stats["allocated_bytes"]["all"]["peak"]
        if peak > self.peak:
            workspace = max(workspace, peak - self.allocated)
        # Unreleasable split blocks can supply workspace without growing the
        # reservation or exceeding a training peak. The initial cache includes
        # those blocks; cumulative batch allocations bound unused cache slack.
        workspace = min(workspace, allocated)
        self.measured_atoms = atoms
        self.bytes_per_atom = workspace / atoms

    def _cached_bytes(self) -> int:
        return sum(
            segment["total_size"] - segment["active_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == self.device and segment["segment_pool_id"] == (0, 0)
        )

    def _available_bytes(self) -> tuple[int, int]:
        """Return available device bytes and reusable default-pool cache bytes."""
        stats = torch.cuda.memory_stats_as_nested_dict(self.device)
        self.device_events = self._device_events(stats)
        free, total = torch.cuda.mem_get_info(self.device)
        reserved = stats["reserved_bytes"]["all"]["current"]
        allowed = int(total * torch.cuda.get_per_process_memory_fraction(self.device))
        return min(free, max(0, allowed - reserved)), self._cached_bytes()

    def _capacity(self, available: int) -> int:
        # The unused fraction covers allocator rounding and moderate variation
        # in frame density; unpredictable workspaces still use OOM backoff.
        capacity = int(0.9 * available / self.bytes_per_atom)
        return max(self.natoms, capacity // self.natoms * self.natoms)

    def limit(self, batch_size: int) -> int:
        if not self.measured_atoms:
            return min(batch_size, self.natoms)
        if not self.bytes_per_atom:
            return batch_size
        free, cached = self._available_bytes()
        target = min(batch_size, self._capacity(free + cached))
        if target > self.measured_atoms and target * self.bytes_per_atom > free:
            # A larger allocation may not fit an existing cached block. Release
            # unused blocks before requesting more pages, rather than relying
            # on the allocator's OOM-triggered cache-flush retry.
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            free, cached = self._available_bytes()
            target = min(target, self._capacity(free + cached))
        return target


class AutoBatchSize(AutoBatchSizeBase):
    """Auto batch size.

    Parameters
    ----------
    initial_batch_size : int, default: 1024
        initial batch size (number of total atoms) when DP_INFER_BATCH_SIZE
        is not set
    factor : float, default: 2.
        increased factor

    Notes
    -----
    Automatic multi-frame CUDA evaluation starts with one calibration frame
    and limits growth using observed workspace and available device memory.
    Explicit ``DP_INFER_BATCH_SIZE`` limits and single-frame calls retain their
    original behavior. Allocator peak counters and memory limits are unchanged.

    """

    def _get_batch_budget(
        self, natoms: int, max_nframes: int | None
    ) -> BatchSizeBudget | None:
        if (
            self._fixed_batch_size
            or max_nframes == 1
            or natoms <= 0
            or env.DEVICE.type != "cuda"
            or not self.is_gpu_available()
            or torch.cuda.get_allocator_backend() != "native"
        ):
            return None
        with torch.cuda.device(env.DEVICE):
            device = torch.cuda.current_device()
            stats = torch.cuda.memory_stats_as_nested_dict(device)
            if isinstance(self._budget, _CudaMemoryBudget) and self._budget.matches(
                device, natoms, stats
            ):
                return self._budget
            torch.cuda.empty_cache()
            return _CudaMemoryBudget(
                device, natoms, torch.cuda.memory_stats_as_nested_dict(device)
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

        if not isinstance(e, RuntimeError):
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
        # https://github.com/deepmodeling/deepmd-kit/issues/4594
        plain_oom_markers = (
            "CUDA out of memory.",
            "CUDA driver error: out of memory",
            "CUDA error: out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED",
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
