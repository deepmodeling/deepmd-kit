# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch LMDB dataset — thin wrapper around framework-agnostic LmdbDataReader."""

import functools
import logging
from collections.abc import (
    Iterator,
)
from typing import (
    Any,
)

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
)

from deepmd.dpmodel.utils.lmdb_data import (
    LmdbBatchIterator,
    LmdbBatchSampler,
    LmdbDataReader,
    LmdbDecodeConfig,
    LmdbTestData,
    collate_lmdb_frames,
    collect_lmdb_sampling_groups,
    compute_block_targets,
    count_group_blocks,
    is_lmdb,
    resolve_per_atom_keys,
    system_block_lookup,
)
from deepmd.env import (
    get_lmdb_num_workers,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "LmdbBatchDataLoader",
    "LmdbDataset",
    "LmdbTestData",
    "_collate_lmdb_batch",
    "is_lmdb",
]


def _collate_lmdb_batch(
    batch: list[dict[str, Any]],
    config: LmdbDecodeConfig,
) -> dict[str, Any]:
    """Collate a list of frame dicts into a torch batch dict.

    Pre-converts per-frame numpy arrays to CPU torch tensors (zero-copy when
    dtype matches) and delegates stacking to the backend-agnostic
    :func:`collate_lmdb_frames`. With torch tensors as input, the shared
    collate yields a torch dict (``sid`` becomes a torch tensor automatically
    via ``array_api_compat``).

    Frames of different atom counts are padded to the batch maximum; the
    padded slots carry the phantom atom type. Frames need not agree on label
    availability: a label only some of them carry is reported unavailable
    for the whole batch.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        Decoded frames to collate.
    config : LmdbDecodeConfig
        Decoder state whose data requirements identify the per-atom fields.

    Returns
    -------
    dict[str, Any]
        One collated batch of CPU tensors.
    """
    per_atom_keys = resolve_per_atom_keys(batch[0], config)
    with torch.device("cpu"):
        torch_frames: list[dict[str, Any]] = []
        for f in batch:
            tf: dict[str, Any] = {}
            for key, val in f.items():
                if key.startswith("find_") or key == "fid" or key == "type":
                    tf[key] = val
                elif val is None:
                    tf[key] = None
                else:
                    tf[key] = torch.as_tensor(val)
            torch_frames.append(tf)
        return collate_lmdb_frames(torch_frames, per_atom_keys)


def _lmdb_batch_to_torch(
    batch: dict[str, Any],
    *,
    pin_memory: bool,
) -> dict[str, Any]:
    """Convert a contiguous NumPy LMDB batch to CPU tensors."""
    converted: dict[str, Any] = {}
    with torch.device("cpu"):
        for key, value in batch.items():
            if key.startswith("find_") or key == "fid" or key == "type":
                converted[key] = value
            elif value is None:
                converted[key] = None
            else:
                tensor = torch.as_tensor(value)
                converted[key] = tensor.pin_memory() if pin_memory else tensor
    return converted


class _LmdbBatchSamplerTorch(Sampler):
    """Torch Sampler adapter around the framework-agnostic LmdbBatchSampler.

    PyTorch DataLoader with batch_sampler expects a Sampler that yields
    lists of indices. This wraps LmdbBatchSampler (or
    DistributedLmdbBatchSampler) to satisfy that.
    """

    def __init__(self, inner: LmdbBatchSampler) -> None:
        self._inner = inner

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._inner

    def __len__(self) -> int:
        return len(self._inner)

    def set_epoch(self, epoch: int) -> None:
        """Forward set_epoch to inner sampler if it supports it."""
        if hasattr(self._inner, "set_epoch"):
            self._inner.set_epoch(epoch)


class LmdbBatchDataLoader:
    """DataLoader-compatible iterable backed by :class:`LmdbBatchIterator`.

    The parent sampler determines batch order. The shared LMDB process pool
    decodes one batch and prefetches its successor, then this adapter converts
    the contiguous NumPy result to pinned CPU tensors.
    """

    def __init__(
        self,
        dataset: "LmdbDataset",
        sampler: Any,
        *,
        pin_memory: bool,
        num_workers: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_sampler = _LmdbBatchSamplerTorch(sampler)
        self.sampler = sampler
        self._pin_memory = pin_memory
        self._batch_iterator = LmdbBatchIterator(
            dataset._reader,
            sampler,
            get_lmdb_num_workers() if num_workers is None else num_workers,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for _ in range(len(self)):
            yield _lmdb_batch_to_torch(
                next(self._batch_iterator),
                pin_memory=self._pin_memory,
            )

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def close(self) -> None:
        """Release this loader's prefetched batch and shared-pool reference."""
        self._batch_iterator.close()

    def __del__(self) -> None:
        """Release the shared-pool reference during interpreter teardown."""
        iterator = getattr(self, "_batch_iterator", None)
        if iterator is not None:
            iterator.close()


class LmdbDataset(Dataset):
    """PyTorch Dataset backed by LMDB via LmdbDataReader.

    Parameters
    ----------
    lmdb_path : str
        Path to the LMDB directory.
    type_map : list[str]
        Global type map from model config.
    batch_size : int or str
        Batch size rule forwarded to :class:`LmdbDataReader`. Supports:

        - ``int``: fixed batch size for every nloc group.
        - ``"auto"`` / ``"auto:N"``: ``ceil(N / nloc)`` per nloc group
          (``N=32`` for bare ``"auto"``).
        - ``"max:N"``: ``max(1, floor(N / nloc))`` per nloc group.
        - ``"filter:N"``: same per-nloc formula as ``"max:N"`` and drops
          every frame whose ``nloc > N`` from the dataset.
        - ``"mix:N"``: mixed-nloc batching to a padded-slot budget of ``N``;
          frames of different atom counts share a batch and the shorter ones
          are padded with phantom atoms.
    auto_prob_style : str, optional
        ``auto_prob`` string used to reweight the original systems.
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str],
        batch_size: int | str = "auto",
        auto_prob_style: str | None = None,
    ) -> None:
        self._reader = LmdbDataReader(lmdb_path, type_map, batch_size)
        self._collate = functools.partial(
            _collate_lmdb_batch, config=self._reader.decode_config
        )

        # Compute block_targets from auto_prob_style if provided. An empty
        # result means the configured probabilities need no reweighting, which
        # is the common case and worth no log line of its own.
        self._block_targets = None
        if auto_prob_style is not None and self._reader.frame_system_ids is not None:
            self._block_targets = compute_block_targets(
                auto_prob_style,
                self._reader.nsystems,
                self._reader.system_nframes,
            )
            if self._block_targets:
                log.info(
                    f"LMDB auto_prob: {len(self._block_targets)} blocks, "
                    f"nsystems={self._reader.nsystems}"
                )

        sampler = LmdbBatchSampler(
            self._reader,
            shuffle=True,
            block_targets=self._block_targets,
        )
        self._batch_sampler = _LmdbBatchSamplerTorch(sampler)

        with torch.device("cpu"):
            self._inner_dataloader = DataLoader(
                self,
                batch_sampler=self._batch_sampler,
                num_workers=0,
                collate_fn=self._collate,
            )

        # Homogeneous dataloaders for make_stat_input, built on first use and
        # discarded whenever new requirements change how frames decode.
        self._nloc_dataloaders: list[DataLoader] | None = None

    def _build_nloc_dataloaders(self) -> None:
        """Build the homogeneous loaders used by model-stat collection."""
        dataloaders: list[DataLoader] = []
        for nloc, indices in collect_lmdb_sampling_groups(self._reader):
            subset = torch.utils.data.Subset(self, indices)
            with torch.device("cpu"):
                dl = DataLoader(
                    subset,
                    batch_size=self._reader.get_batch_size_for_nloc(nloc),
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                    collate_fn=self._collate,
                )
            dataloaders.append(dl)
        self._nloc_dataloaders = dataloaders

    def _get_nloc_dataloaders(self) -> list[DataLoader]:
        """Materialize statistics loaders lazily when none are registered."""
        if self._nloc_dataloaders is None:
            self._build_nloc_dataloaders()
        dataloaders = self._nloc_dataloaders
        if dataloaders is None:
            raise RuntimeError("Failed to initialize LMDB statistics dataloaders")
        return dataloaders

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._reader[index]

    # --- Delegated to reader ---

    @property
    def lmdb_path(self) -> str:
        return self._reader.lmdb_path

    @property
    def nframes(self) -> int:
        return self._reader.nframes

    @property
    def mixed_nloc(self) -> bool:
        """Whether one batch may span several atom counts."""
        return self._reader.mixed_nloc

    @property
    def mixed_type(self) -> bool:
        """LMDB datasets are always mixed_type."""
        return self._reader.mixed_type

    @property
    def batch_size(self) -> int:
        return self._reader.batch_size

    @property
    def type_map(self) -> list[str]:
        return self._reader.type_map

    @property
    def data_requirements(self) -> list[DataRequirementItem]:
        return self._reader.data_requirements

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        self._reader.add_data_requirement(data_requirement)
        # Loaders decode through the registered requirements, so any already
        # built are stale. They are rebuilt on demand, which spares a run
        # whose statistics come from a stat file the work entirely.
        self._nloc_dataloaders = None

    def close(self) -> None:
        """Release parent-process LMDB resources."""
        self._reader.close()

    def __del__(self) -> None:
        """Release parent-process LMDB resources during teardown."""
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()

    def preload_and_modify_all_data_torch(self) -> None:
        """No-op: LMDB reads on demand."""

    def print_summary(self, name: str, prob: Any) -> None:
        self._reader.print_summary(name, prob)
        if self._block_targets:
            reader = self._reader
            # Per-block summary: original vs target frames
            block_lines = []
            total_original = 0
            total_target = 0
            # Pre-compute block_total_actual for proportional scaling
            block_total_actual: list[int] = []
            for sys_ids, target in self._block_targets:
                actual = sum(reader.system_nframes[s] for s in sys_ids)
                block_total_actual.append(actual)
                total_original += actual
                total_target += target
                # Compact range notation: sys[0-146] instead of sys[0,1,2,...,146]
                if len(sys_ids) > 3:
                    sys_str = f"{sys_ids[0]}-{sys_ids[-1]}"
                else:
                    sys_str = ",".join(str(s) for s in sys_ids)
                ratio = target / actual if actual > 0 else 0
                block_lines.append(
                    f"sys[{sys_str}]({len(sys_ids)}sys): "
                    f"{actual}->{target} (x{ratio:.2f})"
                )

            # A whole nloc group's block membership resolves as one indexing
            # operation, which a dataset of 10^8 frames needs it to be. The
            # lookup is the one the sampler allocates its targets with, so
            # both agree on which systems a block claims.
            n_blocks = len(self._block_targets)
            lookup = system_block_lookup(self._block_targets)

            # Compute expanded nloc counts analytically (no actual expansion)
            expanded_nloc_info = {}
            for nloc, indices in sorted(reader.nloc_groups.items()):
                if reader.frame_system_ids is None:
                    expanded_nloc_info[nloc] = len(indices)
                    continue
                counts = count_group_blocks(
                    indices, reader.frame_system_ids, lookup, n_blocks
                )
                expanded = len(indices) - int(counts.sum())
                for blk_idx, (_, blk_target) in enumerate(self._block_targets):
                    n_actual = int(counts[blk_idx])
                    if n_actual == 0:
                        continue
                    bta = block_total_actual[blk_idx]
                    if bta > 0:
                        t = max(round(blk_target * n_actual / bta), n_actual)
                    else:
                        t = n_actual
                    expanded += t
                expanded_nloc_info[nloc] = expanded

            total_expanded = sum(expanded_nloc_info.values())
            n_groups = len(reader.nloc_groups)
            ratio_all = total_expanded / total_original if total_original > 0 else 0

            log.info(
                f"LMDB {name} auto_prob: "
                f"{total_original}->{total_expanded} frames (x{ratio_all:.2f}), "
                f"{n_groups} nloc groups, {len(self._block_targets)} blocks:"
            )
            for bl in block_lines:
                log.info(f"  {bl}")

    def set_noise(self, noise_settings: dict[str, Any]) -> None:
        self._reader.set_noise(noise_settings)

    @property
    def index(self) -> list[int]:
        """Number of batches per logical LMDB dataset."""
        return [self.total_batch]

    @property
    def total_batch(self) -> int:
        return len(self._batch_sampler)

    @property
    def batch_sizes(self) -> list[int]:
        return self._reader.batch_sizes

    # --- PyTorch-specific trainer compatibility ---

    @property
    def systems(self) -> list:
        """One logical system per stack-compatible statistics group."""
        return [self] * len(self._get_nloc_dataloaders())

    @property
    def dataloaders(self) -> list:
        """Homogeneous dataloaders for make_stat_input.

        Each loader draws from one atom count and one label availability, so
        stat collection sees consistent shapes and scalar ``find_*`` flags.
        """
        return self._get_nloc_dataloaders()

    @property
    def sampler_list(self) -> list:
        return [self._batch_sampler]
