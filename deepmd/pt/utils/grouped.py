# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for grouped frame-level property training."""

from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np

from deepmd.utils.data import (
    DataRequirementItem,
)

if TYPE_CHECKING:
    import torch

GROUP_ID_KEY = "group_id"
GROUP_WEIGHT_KEY = "weight"
POOL_MASK_KEY = "pool_mask"


def group_data_requirements() -> list[DataRequirementItem]:
    """Return the auxiliary data fields consumed by ``group_property``."""
    return [
        DataRequirementItem(
            GROUP_ID_KEY,
            ndof=1,
            atomic=False,
            must=False,
            high_prec=False,
            default=0.0,
            dtype=np.int64,
        ),
        DataRequirementItem(
            GROUP_WEIGHT_KEY,
            ndof=1,
            atomic=False,
            must=False,
            high_prec=True,
            default=1.0,
        ),
        DataRequirementItem(
            POOL_MASK_KEY,
            ndof=1,
            atomic=True,
            must=False,
            high_prec=True,
            default=1.0,
            output_natoms_for_type_sel=True,
        ),
    ]


def has_group_requirement(data_requirement: list[DataRequirementItem]) -> bool:
    """Return True when a data requirement asks for grouped auxiliaries."""
    keys = {item["key"] for item in data_requirement}
    return GROUP_ID_KEY in keys and GROUP_WEIGHT_KEY in keys and POOL_MASK_KEY in keys


def normalize_group_id_tensor(group_id: torch.Tensor, nframes: int) -> torch.Tensor:
    """Normalize collated group ids to shape ``(nframes,)``."""
    group_id = group_id.reshape(nframes, -1)
    if group_id.shape[1] != 1:
        raise ValueError("group_id must have one value per frame.")
    return group_id[:, 0].long()


def normalize_weight_tensor(weight: torch.Tensor, nframes: int) -> torch.Tensor:
    """Normalize collated group weights to shape ``(nframes,)``."""
    weight = weight.reshape(nframes, -1)
    if weight.shape[1] != 1:
        raise ValueError("weight must have one value per frame.")
    return weight[:, 0]


def normalize_pool_mask_tensor(
    pool_mask: torch.Tensor, nframes: int, natoms: int
) -> torch.Tensor:
    """Normalize collated pool masks to shape ``(nframes, natoms)``."""
    pool_mask = pool_mask.reshape(nframes, natoms, -1)
    if pool_mask.shape[2] != 1:
        raise ValueError("pool_mask must have one value per atom.")
    return pool_mask[:, :, 0]


def load_group_ids_for_system(system: str | Path) -> np.ndarray | None:
    """Load frame-level group ids from a DeepMD system, if present.

    The returned ids follow DeepMD frame order across sorted ``set.*``
    directories.  Missing data returns ``None`` so callers can fall back to
    ordinary frame batching.
    """
    system_path = Path(system)
    set_dirs = sorted(system_path.glob("set.*"))
    if not set_dirs:
        return None

    chunks: list[np.ndarray] = []
    for set_dir in set_dirs:
        path = set_dir / f"{GROUP_ID_KEY}.npy"
        if not path.is_file():
            return None
        arr = np.asarray(np.load(str(path))).reshape(-1)
        chunks.append(arr.astype(np.int64, copy=False))
    return np.concatenate(chunks) if chunks else None


def _group_frame_indices(group_ids: np.ndarray) -> list[list[int]]:
    """Return frame indices grouped by first-seen group id."""
    if group_ids.ndim != 1:
        raise ValueError(f"{GROUP_ID_KEY} must be 1D; got shape {group_ids.shape}.")
    groups: dict[int, list[int]] = {}
    for frame_idx, group_id in enumerate(group_ids.astype(np.int64, copy=False)):
        groups.setdefault(int(group_id), []).append(frame_idx)
    return list(groups.values())


def _shuffle_group_items(
    group_items: list[list[int]],
    shuffle: bool,
    rng: np.random.Generator | None,
) -> list[list[int]]:
    if not shuffle:
        return list(group_items)
    rng = rng or np.random.default_rng()
    order = rng.permutation(len(group_items))
    return [group_items[int(ii)] for ii in order]


def _pack_group_items(
    group_items: list[list[int]],
    max_frames: int,
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    limit = max(int(max_frames), 1)
    for indices in group_items:
        if current and len(current) + len(indices) > limit:
            batches.append(current)
            current = []
        current.extend(indices)
        if len(current) >= limit:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def grouped_frame_batches(
    group_ids: np.ndarray,
    max_frames: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Pack complete groups into batches without splitting a group."""
    group_items = _shuffle_group_items(
        _group_frame_indices(group_ids), shuffle=shuffle, rng=rng
    )
    return _pack_group_items(group_items, max_frames)


def distributed_grouped_frame_batches(
    group_ids: np.ndarray,
    max_frames: int,
    num_replicas: int,
    rank: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Pack complete groups for one distributed rank.

    Groups, not frames, are assigned to ranks.  Therefore every frame with the
    same ``group_id`` is consumed by exactly one rank/GPU for the epoch.
    """
    num_replicas = int(num_replicas)
    rank = int(rank)
    if num_replicas < 1:
        raise ValueError(f"num_replicas must be >= 1; got {num_replicas}.")
    if rank < 0 or rank >= num_replicas:
        raise ValueError(f"rank must be in [0, {num_replicas}); got rank={rank}.")
    if shuffle and rng is None:
        rng = np.random.default_rng(0)
    group_items = _shuffle_group_items(
        _group_frame_indices(group_ids), shuffle=shuffle, rng=rng
    )
    if len(group_items) < num_replicas:
        raise ValueError(
            "distributed grouped batching requires at least one group per rank; "
            f"got {len(group_items)} groups for {num_replicas} ranks."
        )
    rank_items = group_items[rank::num_replicas]
    return _pack_group_items(rank_items, max_frames)
