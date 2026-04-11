# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch LMDB dataset — thin wrapper around framework-agnostic LmdbDataReader."""

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
from torch.utils.data._utils.collate import (
    collate_tensor_fn,
)

from deepmd.dpmodel.utils.lmdb_data import (
    LmdbDataReader,
    LmdbTestData,
    SameNlocBatchSampler,
    compute_block_targets,
    is_lmdb,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "LmdbDataset",
    "LmdbTestData",
    "_collate_lmdb_batch",
    "is_lmdb",
]


def _collate_lmdb_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of frame dicts into a batch dict.

    All frames in the batch must have the same nloc (enforced by
    SameNlocBatchSampler when mixed_batch=False).

    For mixed_batch=True, this function would need padding + mask.
    Currently raises NotImplementedError for that case.
    """
    if len(batch) > 1:
        atypes = [d.get("atype") for d in batch if d.get("atype") is not None]
        if atypes and any(len(a) != len(atypes[0]) for a in atypes):
            raise NotImplementedError(
                "mixed_batch collation (frames with different atom counts "
                "in the same batch) is not yet supported. "
                "Padding + mask in collate_fn needed."
            )

    example = batch[0]
    result: dict[str, Any] = {}
    for key in example:
        if "find_" in key:
            result[key] = batch[0][key]
        elif key == "fid":
            result[key] = [d[key] for d in batch]
        elif key == "type":
            continue
        elif batch[0][key] is None:
            result[key] = None
        else:
            with torch.device("cpu"):
                result[key] = collate_tensor_fn(
                    [torch.as_tensor(d[key]) for d in batch]
                )
    result["sid"] = torch.tensor([0], dtype=torch.long, device="cpu")
    return result


class _SameNlocBatchSamplerTorch(Sampler):
    """Torch Sampler adapter around the framework-agnostic SameNlocBatchSampler.

    PyTorch DataLoader with batch_sampler expects a Sampler that yields
    lists of indices. This wraps SameNlocBatchSampler (or
    DistributedSameNlocBatchSampler) to satisfy that.
    """

    def __init__(self, inner: SameNlocBatchSampler) -> None:
        self._inner = inner

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._inner

    def __len__(self) -> int:
        return len(self._inner)

    def set_epoch(self, epoch: int) -> None:
        """Forward set_epoch to inner sampler if it supports it."""
        if hasattr(self._inner, "set_epoch"):
            self._inner.set_epoch(epoch)


class LmdbDataset(Dataset):
    """PyTorch Dataset backed by LMDB via LmdbDataReader.

    Parameters
    ----------
    lmdb_path : str
        Path to the LMDB directory.
    type_map : list[str]
        Global type map from model config.
    batch_size : int or str
        Batch size. Supports int, "auto", "auto:N".
    mixed_batch : bool
        If True, allow different nloc in the same batch (future).
        If False (default), use SameNlocBatchSampler.
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str],
        batch_size: int | str = "auto",
        mixed_batch: bool = False,
        auto_prob_style: str | None = None,
    ) -> None:
        self._reader = LmdbDataReader(
            lmdb_path, type_map, batch_size, mixed_batch=mixed_batch
        )

        if mixed_batch:
            # Future: DataLoader with padding collate_fn
            raise NotImplementedError(
                "mixed_batch=True is not yet supported. "
                "Requires padding + mask in collate_fn."
            )

        # Compute block_targets from auto_prob_style if provided
        self._block_targets = None
        if auto_prob_style is not None and self._reader.frame_system_ids is not None:
            self._block_targets = compute_block_targets(
                auto_prob_style,
                self._reader.nsystems,
                self._reader.system_nframes,
            )
            if self._block_targets is not None:
                log.info(
                    f"LMDB auto_prob: {len(self._block_targets)} blocks, "
                    f"nsystems={self._reader.nsystems}"
                )

        # Same-nloc batching: use SameNlocBatchSampler
        sampler = SameNlocBatchSampler(
            self._reader,
            shuffle=True,
            block_targets=self._block_targets,
        )
        self._batch_sampler = _SameNlocBatchSamplerTorch(sampler)

        with torch.device("cpu"):
            self._inner_dataloader = DataLoader(
                self,
                batch_sampler=self._batch_sampler,
                num_workers=0,
                collate_fn=_collate_lmdb_batch,
            )

        # Per-nloc-group dataloaders for make_stat_input.
        # Each group gets its own DataLoader so torch.cat in stat collection
        # only concatenates same-shape tensors.
        self._nloc_dataloaders: list[DataLoader] = []
        for nloc in sorted(self._reader.nloc_groups.keys()):
            indices = self._reader.nloc_groups[nloc]
            subset = torch.utils.data.Subset(self, indices)
            bs = self._reader.get_batch_size_for_nloc(nloc)
            with torch.device("cpu"):
                dl = DataLoader(
                    subset,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                    collate_fn=_collate_lmdb_batch,
                )
            self._nloc_dataloaders.append(dl)

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
    def mixed_batch(self) -> bool:
        return self._reader.mixed_batch

    @property
    def mixed_type(self) -> bool:
        """LMDB datasets are always mixed_type."""
        return self._reader.mixed_type

    @property
    def batch_size(self) -> int:
        return self._reader.batch_size

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        self._reader.add_data_requirement(data_requirement)

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

            # Build sys_id -> block_idx mapping
            sys_to_block: dict[int, int] = {}
            for blk_idx, (sys_ids, _) in enumerate(self._block_targets):
                for sid in sys_ids:
                    sys_to_block[sid] = blk_idx

            # Compute expanded nloc counts analytically (no actual expansion)
            expanded_nloc_info = {}
            for nloc, indices in sorted(reader.nloc_groups.items()):
                if reader.frame_system_ids is None:
                    expanded_nloc_info[nloc] = len(indices)
                    continue
                # Count indices per block in this nloc group
                blk_counts: dict[int, int] = {}
                unassigned = 0
                for idx in indices:
                    sid = reader.frame_system_ids[idx]
                    blk = sys_to_block.get(sid)
                    if blk is not None:
                        blk_counts[blk] = blk_counts.get(blk, 0) + 1
                    else:
                        unassigned += 1
                expanded = unassigned
                for blk_idx, (_, blk_target) in enumerate(self._block_targets):
                    n_actual = blk_counts.get(blk_idx, 0)
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
        return self._reader.index

    @property
    def total_batch(self) -> int:
        return self._reader.total_batch

    @property
    def batch_sizes(self) -> list[int]:
        return self._reader.batch_sizes

    # --- PyTorch-specific trainer compatibility ---

    @property
    def systems(self) -> list:
        """One 'system' per nloc group for stat collection compatibility."""
        return [self] * len(self._nloc_dataloaders)

    @property
    def dataloaders(self) -> list:
        """Per-nloc-group dataloaders for make_stat_input.

        Each dataloader yields batches with uniform nloc, so torch.cat
        in stat collection only concatenates same-shape tensors.
        """
        return self._nloc_dataloaders

    @property
    def sampler_list(self) -> list:
        return [self._batch_sampler]
