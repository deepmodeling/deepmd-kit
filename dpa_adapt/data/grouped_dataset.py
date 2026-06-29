# SPDX-License-Identifier: LGPL-3.0-or-later
"""Grouped descriptor dataset for shared-target structures."""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Iterable

import dpdata
import numpy as np

from dpa_adapt.data.aggregation import aggregate_weighted_groups
from dpa_adapt.data.errors import DPADataError
from dpa_adapt.data.loader import _get_source, _resolve_label_key, load_data
from dpa_adapt.finetuner import load_or_extract


class GroupedDataset:
    """Aggregate per-frame descriptors into one sample per group id."""

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

        for system in self.systems:
            source = _get_source(system)
            if source is None:
                raise DPADataError(
                    "Grouped input must come from deepmd/npy directories so "
                    "set.*/category.npy and set.*/weight.npy can be read."
                )
            source_path = Path(source)
            system_frames = _read_system_group_rows(source_path, self.target_key)
            for group_id, weight, label in system_frames:
                group_ids.append(group_id)
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
    """Return True when any resolved system directory has category.npy."""
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
    return sorted(
        child for child in path.iterdir() if child.is_dir() and _is_system_dir(child)
    )


def _is_system_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("set.*"))


def _system_has_marker(path: Path) -> bool:
    return any(
        set_dir.joinpath("category.npy").is_file()
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

    system_group_id: int | None = None
    for set_dir in set_dirs:
        category_path = set_dir / "category.npy"
        weight_path = set_dir / "weight.npy"
        label_path = set_dir / f"{target_key}.npy"
        missing = [
            str(p)
            for p in (category_path, weight_path, label_path)
            if not p.is_file()
        ]
        if missing:
            raise DPADataError(f"Grouped input is missing required files: {missing}.")

        category = np.asarray(np.load(str(category_path)))
        if category.ndim != 1:
            raise DPADataError(
                f"{category_path} has shape {category.shape}; expected (n_atoms,)."
            )
        unique = np.unique(category.astype(np.int64, copy=False))
        if unique.size != 1:
            raise DPADataError(
                f"{category_path} contains {unique.size} ids; expected one id "
                "per system."
            )
        group_id = int(unique[0])
        if system_group_id is None:
            system_group_id = group_id
        elif group_id != system_group_id:
            raise DPADataError(
                f"{source_path} uses multiple group ids across set.* directories; "
                "expected one id per system."
            )

        weight = np.asarray(np.load(str(weight_path)), dtype=float)
        if weight.ndim != 1:
            raise DPADataError(
                f"{weight_path} has shape {weight.shape}; expected (n_frames,)."
            )
        labels = np.asarray(np.load(str(label_path)))
        if labels.shape[0] != weight.shape[0]:
            raise DPADataError(
                f"{label_path} has {labels.shape[0]} rows but {weight_path} has "
                f"{weight.shape[0]}."
            )
        for frame_idx in range(weight.shape[0]):
            rows.append(
                (group_id, float(weight[frame_idx]), np.asarray(labels[frame_idx]))
            )
    return rows
