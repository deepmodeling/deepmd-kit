# SPDX-License-Identifier: LGPL-3.0-or-later
"""Grouped descriptor dataset for shared-target structures."""

from __future__ import (
    annotations,
)

import glob as _glob
import os
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)

import dpdata
import numpy as np

from dpa_adapt.data.errors import (
    DPADataError,
)
from dpa_adapt.data.loader import (
    _get_source,
    _resolve_label_key,
    load_data,
)
from dpa_adapt.grouped._aggregation import (
    aggregate_weighted_groups,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )


def load_or_extract(*args: object, **kwargs: object) -> np.ndarray:
    """Delegate to finetuner.load_or_extract without a module import cycle."""
    import importlib

    finetuner = importlib.import_module("dpa_adapt.finetuner")
    return finetuner.load_or_extract(*args, **kwargs)


class GroupedDataset:
    """Aggregate per-frame descriptors into one row per group id."""

    def __init__(
        self,
        data: str | Path | list[str | Path | dpdata.System],
        pretrained: str,
        model_branch: str | None = None,
        type_map: list[str] | tuple[str, ...] | None = None,
        target_key: str = "property",
        fmt: str | None = None,
        cache: bool = True,
    ) -> None:
        self.data = data
        self.pretrained = pretrained
        self.model_branch = model_branch
        self.type_map = list(type_map) if type_map else None
        self.target_key = _resolve_label_key(target_key)
        self.fmt = fmt
        self.cache = cache

        self.systems = _load_groupable_systems(data, fmt=fmt)
        self._embeddings, self._labels = self._build()

    def get_embeddings(self) -> np.ndarray:
        return self._embeddings

    def get_labels(self) -> np.ndarray:
        return self._labels

    def _build(self) -> tuple[np.ndarray, np.ndarray]:
        group_ids: list[int] = []
        weights: list[float] = []
        labels: list[np.ndarray] = []

        next_group_id = 0
        for system in self.systems:
            source = _get_source(system)
            if source is None:
                raise DPADataError(
                    "Assembly input must come from deepmd/npy directories so "
                    "set.*/group_id.npy can be read."
                )
            source_path = Path(source)
            system_frames = _read_system_group_rows(source_path, self.target_key)
            # group_id.npy is scoped to one DeepMD system.  Many assembly writers
            # naturally use group_id=0 in every system, so remap each system's
            # local ids into a process-wide id space before offline aggregation.
            local_to_global: dict[int, int] = {}
            for local_group_id, weight, label in system_frames:
                if local_group_id not in local_to_global:
                    local_to_global[local_group_id] = next_group_id
                    next_group_id += 1
                group_ids.append(local_to_global[local_group_id])
                weights.append(float(weight))
                labels.append(np.asarray(label))

        if not group_ids:
            raise DPADataError("Grouped input contains no frames to aggregate.")

        descriptors = load_or_extract(
            self.systems,
            pretrained=self.pretrained,
            model_branch=self.model_branch,
            pooling="mean",
            cache=self.cache,
            type_map=self.type_map,
        )
        if descriptors.shape[0] != len(group_ids):
            raise DPADataError(
                f"Descriptor rows ({descriptors.shape[0]}) do not match grouped "
                f"frame rows ({len(group_ids)})."
            )

        embeddings, label_arr, _ = aggregate_weighted_groups(
            descriptors,
            np.asarray(group_ids, dtype=np.int64),
            np.asarray(weights, dtype=float),
            np.asarray(labels),
        )
        return embeddings, label_arr


def has_grouped_markers(data: object) -> bool:
    """Return True when any resolved system directory has group_id.npy."""
    return any(_system_has_marker(path) for path in _candidate_paths(data))


def _load_groupable_systems(
    data: str | Path | list[str | Path | dpdata.System],
    fmt: str | None = None,
) -> list[dpdata.System]:
    if isinstance(data, list):
        systems: list[dpdata.System] = []
        for item in data:
            systems.extend(_load_groupable_systems(item, fmt=fmt))
        return systems
    if isinstance(data, (dpdata.System, dpdata.LabeledSystem)):
        return [data]

    paths = _candidate_paths(data)
    if not paths:
        raise DPADataError(f"No grouped system directories found under {data!r}.")
    systems: list[dpdata.System] = []
    for path in paths:
        systems.extend(load_data(str(path), fmt=fmt))
    return systems


def _candidate_paths(data: object) -> list[Path]:
    if isinstance(data, list):
        paths: list[Path] = []
        for item in data:
            paths.extend(_candidate_paths(item))
        return _unique_paths(paths)
    if isinstance(data, (dpdata.System, dpdata.LabeledSystem)):
        source = _get_source(data)
        return [Path(source)] if source is not None else []
    if not isinstance(data, (str, Path)):
        return []

    raw = str(data)
    if _glob.has_magic(raw):
        return _unique_paths(
            path
            for match in sorted(_glob.glob(raw))
            for path in _paths_from_directory(Path(match))
        )
    return _paths_from_directory(Path(raw))


def _paths_from_directory(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if _is_system_dir(path):
        return [path]
    if not path.is_dir():
        return []
    # Recurse arbitrarily deep so discovery matches ``mark_groups`` /
    # ``_walk_systems`` (which uses ``os.walk``); scanning only immediate
    # children would silently drop systems nested more than one level down.
    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(path):
        candidate = Path(dirpath)
        if _is_system_dir(candidate):
            found.append(candidate)
            # Do not descend into this system's own set.* directories.
            dirnames[:] = [d for d in dirnames if not d.startswith("set.")]
    return sorted(found)


def _is_system_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("set.*"))


def _system_has_marker(path: Path) -> bool:
    return any(
        set_dir.joinpath("group_id.npy").is_file()
        for set_dir in sorted(path.glob("set.*"))
    )


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def _read_system_group_rows(
    source_path: Path, target_key: str
) -> list[tuple[int, float, np.ndarray]]:
    rows: list[tuple[int, float, np.ndarray]] = []
    set_dirs = sorted(source_path.glob("set.*"))
    if not set_dirs:
        raise DPADataError(f"No set.* directories found in {source_path}.")

    for set_dir in set_dirs:
        group_id_path = set_dir / "group_id.npy"
        weight_path = set_dir / "weight.npy"
        label_path = set_dir / f"{target_key}.npy"
        missing = [str(p) for p in (group_id_path, label_path) if not p.is_file()]
        if missing:
            raise DPADataError(f"Grouped input is missing required files: {missing}.")

        group_ids = np.asarray(np.load(str(group_id_path)), dtype=np.int64).reshape(-1)
        labels = np.asarray(np.load(str(label_path)))
        n_frames = labels.shape[0]
        if weight_path.is_file():
            weight = np.asarray(np.load(str(weight_path)), dtype=float).reshape(-1)
        else:
            weight = np.ones((n_frames,), dtype=float)
        if group_ids.shape != (n_frames,):
            raise DPADataError(
                f"{group_id_path} has shape {group_ids.shape}; expected ({n_frames},)."
            )
        if weight.shape != (n_frames,):
            raise DPADataError(
                f"{weight_path} has shape {weight.shape}; expected ({n_frames},)."
            )
        for frame_idx in range(n_frames):
            rows.append(
                (
                    int(group_ids[frame_idx]),
                    float(weight[frame_idx]),
                    np.asarray(labels[frame_idx]),
                )
            )
    return rows
