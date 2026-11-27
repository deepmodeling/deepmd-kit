# SPDX-License-Identifier: LGPL-3.0-or-later
"""Framework-independent LMDB data-system adapter for NumPy batch consumers."""

from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.lmdb_data import (
    DistributedLmdbBatchSampler,
    LmdbBatchIterator,
    LmdbBatchSampler,
    LmdbDataReader,
    LmdbTestData,
    LmdbTestDataNlocView,
    collect_lmdb_sampling_groups,
    compute_block_targets,
)
from deepmd.env import (
    get_lmdb_num_workers,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

__all__ = ["LmdbDataSystem"]


class LmdbDataSystem:
    """LMDB-backed data system with prefetching and grouped statistics.

    Exposes the common NumPy trainer data-system surface:
    ``get_batch(sys_idx=None)``, ``add_data_requirements(list)``,
    ``get_nsystems()``, and the ``nbatches``/``sys_probs`` pair from which the
    trainer derives an epoch length. The whole LMDB counts as one logical
    system. Internally uses :class:`LmdbDataReader` for I/O and
    :class:`LmdbBatchSampler`, or its distributed wrapper, to draw batches.
    Statistics use a separate logical-system view in which every
    ``(nloc, label-availability)`` group is sampled independently, matching
    the training sampler without changing the identity of the LMDB as one
    training dataset.

    Parameters
    ----------
    lmdb_path
        Path to the LMDB directory.
    type_map
        Global type map from the model config.
    batch_size
        Batch size spec; ``int``, ``"auto"``, ``"auto:N"``, ``"max:N"``,
        ``"filter:N"``, or ``"mix:N"`` for mixed-nloc batching.
    auto_prob_style
        Optional ``auto_prob`` string (e.g. ``"prob_sys_size"``) for
        per-system reweighting via :func:`compute_block_targets`.
    seed
        Optional seed for the shuffle in :class:`LmdbBatchSampler`.
    num_workers
        Number of LMDB decoder worker processes. ``None`` selects the
        hardware-aware default; zero or one disables multiprocessing.
    rank
        Rank of this process in distributed training.
    world_size
        Number of distributed training processes. Values greater than one
        select :class:`DistributedLmdbBatchSampler`.
    batch_multiple
        Pad each local sampling pass to this multiple of batches. Padding
        repeats batches from the start of the epoch; it never draws from the
        following epoch. Pass lengths include these repeated batches.
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str],
        batch_size: int | str = "auto",
        auto_prob_style: str | None = None,
        seed: int | None = None,
        num_workers: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        batch_multiple: int = 1,
    ) -> None:
        self._reader = LmdbDataReader(lmdb_path, type_map, batch_size)
        self._lmdb_path = lmdb_path
        self._test_data: LmdbTestData | None = None
        self._test_views: list[LmdbTestDataNlocView] | None = None

        block_targets = None
        if auto_prob_style is not None and self._reader.frame_system_ids is not None:
            block_targets = compute_block_targets(
                auto_prob_style,
                self._reader.nsystems,
                self._reader.system_nframes,
            )

        if world_size > 1:
            self._sampler: LmdbBatchSampler | DistributedLmdbBatchSampler = (
                DistributedLmdbBatchSampler(
                    self._reader,
                    rank=rank,
                    world_size=world_size,
                    shuffle=True,
                    seed=seed,
                    block_targets=block_targets,
                    pad_to_multiple=batch_multiple,
                )
            )
        else:
            self._sampler = LmdbBatchSampler(
                self._reader,
                shuffle=True,
                seed=seed,
                block_targets=block_targets,
                pad_to_multiple=batch_multiple,
            )
        self._refresh_stat_groups()
        num_workers = (
            get_lmdb_num_workers() if num_workers is None else int(num_workers)
        )
        self._batch_iterator = LmdbBatchIterator(
            self._reader,
            self._sampler,
            num_workers,
        )

    def _refresh_stat_groups(self) -> None:
        """Build statistical systems from stack-compatible reader groups."""
        self._stat_groups = collect_lmdb_sampling_groups(self._reader)
        self._stat_offsets = [0] * len(self._stat_groups)

    # ------------------------------------------------------------------
    # NumPy trainer surface
    # ------------------------------------------------------------------

    def use_ragged_batches(self, ragged: bool) -> None:
        """Select the layout :meth:`get_batch` delivers.

        See :meth:`deepmd.dpmodel.utils.lmdb_data.LmdbDataReader.use_ragged_batches`.
        The trainer calls this once it knows whether the model reads a flat
        node axis, before training draws its first batch.

        Parameters
        ----------
        ragged : bool
            Whether to concatenate frames instead of padding them.
        """
        self._reader.use_ragged_batches(ragged)

    def get_batch(self, sys_idx: int | None = None) -> dict[str, Any]:
        """Return one batch as a numpy dict.

        ``sys_idx`` is accepted for API compatibility but ignored: per-system
        sampling is baked into ``block_targets`` at sampler construction.
        """
        del sys_idx
        return next(self._batch_iterator)

    def state_dict(self) -> dict[str, Any]:
        """Return the next training-batch cursor, excluding statistical reads."""
        return self._batch_iterator.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore training progress without drawing or decoding any batch."""
        self._batch_iterator.load_state_dict(state)

    def get_model_sampling_size(self) -> int:
        """Count retained frames, matching native PyTorch LMDB task weights."""
        return len(self._reader)

    @property
    def uniform_sampling_mass(self) -> float:
        """Return the queued batch's systems-per-frame selection weight."""
        return self._batch_iterator.uniform_sampling_mass

    @property
    def uniform_nominal_mass(self) -> float:
        """Return the pending pass's mean systems-per-frame selection weight."""
        return self._batch_iterator.uniform_nominal_mass

    def get_epoch_batch_counts(self, world_size: int) -> np.ndarray:
        """Return the actual local workload after distributed batch padding."""
        return np.asarray([len(self._sampler)], dtype=np.float64)

    def get_type_map(self) -> list[str]:
        return self.type_map

    @property
    def system_dirs(self) -> list[str]:
        return [self._reader.lmdb_path]

    def get_stat_batch(self, sys_idx: int) -> dict[str, Any]:
        """Return one batch from a homogeneous statistical system.

        Parameters
        ----------
        sys_idx : int
            Index into the per-atom-count groups.

        Returns
        -------
        dict[str, Any]
            A collated NumPy batch whose frames have one atom count.

        Raises
        ------
        IndexError
            If ``sys_idx`` does not identify an available statistical group.

        Notes
        -----
        The batch is rectangular whatever layout training uses: output
        statistics accumulate over an ``(nf, nloc, ...)`` axis. A group is
        uniform in atom count, so that layout pads nothing.
        """
        if not 0 <= sys_idx < len(self._stat_groups):
            raise IndexError(
                f"Statistical system index {sys_idx} is out of range for "
                f"{len(self._stat_groups)} homogeneous groups."
            )

        nloc, group_indices = self._stat_groups[sys_idx]
        batch_size = self._reader.get_batch_size_for_nloc(nloc)
        start = self._stat_offsets[sys_idx]
        if start >= len(group_indices):
            start = 0
        stop = min(start + batch_size, len(group_indices))
        self._stat_offsets[sys_idx] = stop
        return self._reader.decode_batch(group_indices[start:stop], ragged=False)

    def get_stat_nsystems(self) -> int:
        """Return the number of homogeneous statistical systems."""
        return len(self._stat_groups)

    def get_stat_numb_batches(self, sys_idx: int) -> int:
        """Return the available batch count for one statistical system."""
        if not 0 <= sys_idx < len(self._stat_groups):
            raise IndexError(
                f"Statistical system index {sys_idx} is out of range for "
                f"{len(self._stat_groups)} homogeneous groups."
            )
        nloc, group_indices = self._stat_groups[sys_idx]
        nframes = len(group_indices)
        batch_size = self._reader.get_batch_size_for_nloc(nloc)
        return (nframes + batch_size - 1) // batch_size

    def add_data_requirements(
        self, data_requirement: list[DataRequirementItem]
    ) -> None:
        self._reader.add_data_requirement(data_requirement)
        self._refresh_stat_groups()
        if self._test_data is not None:
            self._test_data.add_data_requirement(data_requirement)
            self._test_views = None

    def set_frame_transform(
        self,
        transform: Callable[[dict[str, Any], int], dict[str, Any]] | None,
    ) -> None:
        """Install a per-frame transform on the underlying reader."""
        self._reader.set_frame_transform(transform)

    def close(self) -> None:
        """Cancel prefetched work and release decoder processes."""
        iterator = getattr(self, "_batch_iterator", None)
        if iterator is not None:
            iterator.close()
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()
        self._test_views = None
        self._test_data = None

    def __del__(self) -> None:
        """Release worker processes during interpreter teardown."""
        self.close()

    def get_nsystems(self) -> int:
        """Return one logical LMDB training dataset."""
        return 1

    def get_batch_pass_length(self) -> int:
        """Return the local pass length belonging to the last returned batch."""
        return self._batch_iterator.batch_pass_length

    def get_batch_size(self) -> list[int]:
        """Return the nominal per-system batch size."""
        return self._reader.batch_sizes

    def get_ntypes(self) -> int:
        """Return the number of types in the model-side type map."""
        return len(self.type_map)

    @property
    def nbatches(self) -> list[int]:
        """Return the global batch count of one full pass."""
        return [self._sampler.total_batches]

    @property
    def sys_probs(self) -> list[float]:
        """Return the sampling probability of each logical system."""
        return [1.0]

    # ------------------------------------------------------------------
    # Misc forwarders
    # ------------------------------------------------------------------

    @property
    def type_map(self) -> list[str]:
        return self._reader.type_map

    @property
    def mixed_type(self) -> bool:
        return True

    @property
    def data_systems(self) -> list[LmdbTestDataNlocView]:
        """Expose homogeneous validation views without decoding whole datasets.

        Each view supports ``iter_test`` for bounded-memory validation. Label
        availability is uniform within a view, so scalar ``find_*`` flags
        retain the same meaning as in the ordinary data-system interface.
        """
        if self._test_data is None:
            self._test_data = LmdbTestData(
                self._lmdb_path,
                type_map=self.type_map,
                shuffle_test=False,
            )
            self._test_data.add_data_requirement(self._reader.data_requirements)
        if self._test_views is None:
            self._test_views = [
                LmdbTestDataNlocView(self._test_data, nloc, indices)
                for nloc, group in self._test_data.nloc_groups.items()
                for indices in self._test_data.availability_groups(group)
            ]
        return self._test_views

    def print_summary(self, name: str, prob: Any = None) -> None:
        self._reader.print_summary(name, prob)
