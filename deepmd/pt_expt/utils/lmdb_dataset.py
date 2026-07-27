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
    LmdbDataReader,
    SameNlocBatchSampler,
    collate_lmdb_frames,
    compute_block_targets,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

__all__ = ["LmdbDataSystem"]


class LmdbDataSystem:
    """LMDB-backed data system for pt_expt.

    Exposes the small surface that pt_expt's trainer touches:
    ``get_batch(sys_idx=None)``, ``add_data_requirements(list)``, and
    ``get_nsystems()``. Internally uses :class:`LmdbDataReader` for I/O and
    :class:`SameNlocBatchSampler` to draw same-nloc batches. Statistics use a
    separate logical-system view in which every ``nloc`` group is sampled
    independently, matching the PyTorch DataLoader adapter without changing
    the identity of the LMDB as one training dataset.

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
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str],
        batch_size: int | str = "auto",
        auto_prob_style: str | None = None,
        seed: int | None = None,
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

        self._sampler = SameNlocBatchSampler(
            self._reader,
            shuffle=True,
            seed=seed,
            block_targets=block_targets,
        )
        self._iter = iter(self._sampler)
        self._stat_nlocs = tuple(sorted(self._reader.nloc_groups))
        self._stat_offsets = [0] * len(self._stat_nlocs)

    # ------------------------------------------------------------------
    # pt_expt trainer surface
    # ------------------------------------------------------------------

    def get_batch(self, sys_idx: int | None = None) -> dict[str, Any]:
        """Return one batch as a numpy dict.

        ``sys_idx`` is accepted for API compatibility but ignored: per-system
        sampling is baked into ``block_targets`` at sampler construction.
        """
        del sys_idx
        try:
            indices = next(self._iter)
        except StopIteration:
            self._iter = iter(self._sampler)
            indices = next(self._iter)
        return self._collate_indices(indices)

    def get_stat_batch(self, sys_idx: int) -> dict[str, Any]:
        """Return one batch from a fixed-``nloc`` statistical system.

        Parameters
        ----------
        sys_idx : int
            Index into the sorted ``nloc`` groups.

        Returns
        -------
        dict[str, Any]
            A collated NumPy batch whose frames have one atom count.

        Raises
        ------
        IndexError
            If ``sys_idx`` does not identify an available ``nloc`` group.
        """
        if not 0 <= sys_idx < len(self._stat_nlocs):
            raise IndexError(
                f"Statistical system index {sys_idx} is out of range for "
                f"{len(self._stat_nlocs)} nloc groups."
            )

        nloc = self._stat_nlocs[sys_idx]
        group_indices = self._reader.nloc_groups[nloc]
        batch_size = self._reader.get_batch_size_for_nloc(nloc)
        start = self._stat_offsets[sys_idx]
        if start >= len(group_indices):
            start = 0
        stop = min(start + batch_size, len(group_indices))
        self._stat_offsets[sys_idx] = stop
        return self._collate_indices(group_indices[start:stop])

    def get_stat_nsystems(self) -> int:
        """Return the number of fixed-``nloc`` statistical systems."""
        return len(self._stat_nlocs)

    def get_stat_numb_batches(self, sys_idx: int) -> int:
        """Return the available batch count for one statistical system."""
        if not 0 <= sys_idx < len(self._stat_nlocs):
            raise IndexError(
                f"Statistical system index {sys_idx} is out of range for "
                f"{len(self._stat_nlocs)} nloc groups."
            )
        nloc = self._stat_nlocs[sys_idx]
        nframes = len(self._reader.nloc_groups[nloc])
        batch_size = self._reader.get_batch_size_for_nloc(nloc)
        return (nframes + batch_size - 1) // batch_size

    def _collate_indices(self, indices: list[int]) -> dict[str, Any]:
        """Load and collate the requested dataset indices."""
        frames = [self._reader[int(i)] for i in indices]
        return collate_lmdb_frames(frames)

    def add_data_requirements(
        self, data_requirement: list[DataRequirementItem]
    ) -> None:
        self._reader.add_data_requirement(data_requirement)

    def get_nsystems(self) -> int:
        """Return one logical LMDB training dataset."""
        return 1

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
