# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMDB data adapter for the pt_expt backend.

pt_expt does not use ``torch.utils.data.DataLoader``; its trainer calls
``data_sys.get_batch()`` directly and expects a numpy dict in the
``DeepmdDataSystem`` shape (the shape consumed by
``deepmd.dpmodel.utils.batch.normalize_batch``). This module provides a thin
wrapper around the framework-agnostic :class:`LmdbDataReader` that satisfies
that interface.
"""

from typing import (
    Any,
)

from deepmd.dpmodel.utils.lmdb_data import (
    DistributedSameNlocBatchSampler,
    LmdbBatchIterator,
    LmdbDataReader,
    SameNlocBatchSampler,
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
    """LMDB-backed data system for pt_expt.

    Exposes the small surface that pt_expt's trainer touches:
    ``get_batch(sys_idx=None)``, ``add_data_requirements(list)``,
    ``get_nsystems()``, and the ``nbatches``/``sys_probs`` pair from which the
    trainer derives an epoch length. The whole LMDB counts as one logical
    system. Internally uses :class:`LmdbDataReader` for I/O and
    :class:`SameNlocBatchSampler`, or its distributed wrapper, to draw
    same-nloc batches. Statistics use a separate logical-system view in which
    every ``(nloc, label-availability)`` group is sampled independently,
    matching the training sampler without changing the identity of the LMDB as
    one training dataset.

    Parameters
    ----------
    lmdb_path
        Path to the LMDB directory.
    type_map
        Global type map from the model config.
    batch_size
        Batch size spec; ``int``, ``"auto"``, or ``"auto:N"``.
    auto_prob_style
        Optional ``auto_prob`` string (e.g. ``"prob_sys_size"``) for
        per-system reweighting via :func:`compute_block_targets`.
    seed
        Optional seed for the shuffle in :class:`SameNlocBatchSampler`.
    num_workers
        Number of LMDB decoder worker processes. ``None`` selects the
        hardware-aware default; zero or one disables multiprocessing.
    rank
        Rank of this process in distributed training.
    world_size
        Number of distributed training processes. Values greater than one
        select :class:`DistributedSameNlocBatchSampler`.
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
    ) -> None:
        self._reader = LmdbDataReader(
            lmdb_path, type_map, batch_size, mixed_batch=False
        )

        block_targets = None
        if auto_prob_style is not None and self._reader.frame_system_ids is not None:
            block_targets = compute_block_targets(
                auto_prob_style,
                self._reader.nsystems,
                self._reader.system_nframes,
            )

        if world_size > 1:
            distributed_sampler = DistributedSameNlocBatchSampler(
                self._reader,
                rank=rank,
                world_size=world_size,
                shuffle=True,
                seed=seed,
                block_targets=block_targets,
            )
            self._sampler = distributed_sampler
        else:
            sampler = SameNlocBatchSampler(
                self._reader,
                shuffle=True,
                seed=seed,
                block_targets=block_targets,
            )
            self._sampler = sampler
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
        """Rebuild statistical systems from the training sampler's groups."""
        self._stat_groups = collect_lmdb_sampling_groups(self._reader)
        self._stat_offsets = [0] * len(self._stat_groups)

    # ------------------------------------------------------------------
    # pt_expt trainer surface
    # ------------------------------------------------------------------

    def get_batch(self, sys_idx: int | None = None) -> dict[str, Any]:
        """Return one batch as a numpy dict.

        ``sys_idx`` is accepted for API compatibility but ignored: per-system
        sampling is baked into ``block_targets`` at sampler construction.
        """
        del sys_idx
        return next(self._batch_iterator)

    def get_stat_batch(self, sys_idx: int) -> dict[str, Any]:
        """Return one batch from a homogeneous statistical system.

        Parameters
        ----------
        sys_idx : int
            Index into the ``(nloc, label-availability)`` groups.

        Returns
        -------
        dict[str, Any]
            A collated NumPy batch whose frames have one atom count.

        Raises
        ------
        IndexError
            If ``sys_idx`` does not identify an available statistical group.
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
        return self._reader.decode_batch(group_indices[start:stop])

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
        # Batches are partitioned by label availability. The sampler derives
        # the partition on its first draw; only the distributed batch count is
        # cached, so it is refreshed after the requirements change.
        self._reader.add_data_requirement(data_requirement)
        self._refresh_stat_groups()
        if isinstance(self._sampler, DistributedSameNlocBatchSampler):
            self._sampler.refresh_batch_count()

    def close(self) -> None:
        """Cancel prefetched work and release decoder processes."""
        iterator = getattr(self, "_batch_iterator", None)
        if iterator is not None:
            iterator.close()
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()

    def __del__(self) -> None:
        """Release worker processes during interpreter teardown."""
        self.close()

    def get_nsystems(self) -> int:
        """Return one logical LMDB training dataset."""
        return 1

    @property
    def nbatches(self) -> list[int]:
        """Return the global batch count of one full pass."""
        if isinstance(self._sampler, DistributedSameNlocBatchSampler):
            return [self._sampler.total_batches]
        return [len(self._sampler)]

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

    def print_summary(self, name: str, prob: Any = None) -> None:
        self._reader.print_summary(name, prob)
