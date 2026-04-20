# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMDB data adapter for the pt_expt backend.

pt_expt does not use ``torch.utils.data.DataLoader``; its trainer calls
``data_sys.get_batch()`` directly and expects a numpy dict in the
``DeepmdDataSystem`` shape (the shape consumed by
``deepmd.dpmodel.utils.batch.normalize_batch``). This module provides a thin
wrapper around the framework-agnostic :class:`LmdbDataReader` that satisfies
that interface.
"""

import logging
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

log = logging.getLogger(__name__)

__all__ = ["LmdbDataSystem"]


class LmdbDataSystem:
    """LMDB-backed data system for pt_expt.

    Exposes the small surface that pt_expt's trainer touches:
    ``get_batch(sys_idx=None)``, ``add_data_requirements(list)``, and
    ``get_nsystems()``. Internally uses :class:`LmdbDataReader` for I/O and
    :class:`SameNlocBatchSampler` to draw same-nloc batches.

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
        frames = [self._reader[int(i)] for i in indices]
        return collate_lmdb_frames(frames)

    def add_data_requirements(
        self, data_requirement: list[DataRequirementItem]
    ) -> None:
        self._reader.add_data_requirement(data_requirement)

    def get_nsystems(self) -> int:
        """Return 1: pt_expt's stat collection treats LMDB as a single system.

        Per-system sampling within the LMDB is handled by
        ``SameNlocBatchSampler`` + ``block_targets``.
        """
        return 1

    # ------------------------------------------------------------------
    # Misc forwarders
    # ------------------------------------------------------------------

    @property
    def type_map(self) -> list[str]:
        return self._reader._type_map

    @property
    def mixed_type(self) -> bool:
        return True

    def print_summary(self, name: str, prob: Any = None) -> None:
        self._reader.print_summary(name, prob)
