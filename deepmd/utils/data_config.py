# SPDX-License-Identifier: LGPL-3.0-or-later
"""Format-independent configuration of training data groups."""

import logging
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    TypeAlias,
)

from deepmd.utils.data_path import (
    DatasetFormat,
    is_lmdb,
    process_systems,
)
from deepmd.utils.data_sampling import (
    group_probabilities,
)
from deepmd.utils.path import (
    DPPath,
)

DataConfig: TypeAlias = Mapping[str, Any] | list[Mapping[str, Any]]
BatchSize: TypeAlias = int | str | list[int]
log = logging.getLogger(__name__)


def check_type_map_consistency(type_maps: Sequence[list[str] | None]) -> list[str]:
    """Return the longest mutually consistent type map."""
    result: list[str] = []
    for type_map in type_maps:
        if type_map is None:
            continue
        common = min(len(result), len(type_map))
        if result[:common] != type_map[:common]:
            raise RuntimeError(
                f"Inconsistent type map: {result!s} {type_map!s} in different data systems. "
                "If you didn't set model/type_map, please set it, "
                "since the type map of the model cannot be decided by data."
            )
        if len(type_map) > len(result):
            result = list(type_map)
    return result


def data_group_configs(config: DataConfig) -> list[Mapping[str, Any]]:
    """View a legacy configuration as one group without mutating its input."""
    groups = [config] if isinstance(config, Mapping) else config
    if not isinstance(groups, list) or not groups:
        raise ValueError(
            "Data configuration must be a dictionary or a nonempty list of dictionaries."
        )
    if any(not isinstance(group, Mapping) for group in groups):
        raise ValueError("Every data group must be a dictionary.")
    group_probabilities([1] * len(groups), [group.get("prob") for group in groups])
    return groups


@dataclass(frozen=True)
class DataGroupConfig:
    """One configured group after recursive dataset discovery.

    ``prob`` controls the whole group's batch-selection probability.
    ``auto_prob`` and ``sys_probs`` retain their native within-group
    semantics. An LMDB's internal system numbering is local to that file;
    custom ``auto_prob`` rules therefore require a single LMDB source.
    """

    systems: tuple[str, ...]
    batch_size: BatchSize
    auto_prob: str | None
    sys_probs: list[float] | None
    prob: float | None


@dataclass(frozen=True)
class DataSourceConfig:
    """One native batch stream within a configured group."""

    format: DatasetFormat
    systems: tuple[str, ...]
    batch_size: BatchSize


def resolve_data_sources(group: DataGroupConfig) -> list[DataSourceConfig]:
    """Partition a group into native readers after whole-system filtering.

    NumPy systems share a native reader; each LMDB owns one. NumPy atom-count
    filtering precedes this partition so an empty NumPy subset does not reject
    a group with retained LMDB data. LMDB filtering remains frame-local.
    """
    numpy_indices = [
        index for index, path in enumerate(group.systems) if not is_lmdb(path)
    ]
    numpy_batch_size = group.batch_size
    if isinstance(numpy_batch_size, str) and numpy_batch_size.startswith("filter:"):
        limit = int(numpy_batch_size.split(":")[1])
        retained = [
            index
            for index in numpy_indices
            if (DPPath(group.systems[index]) / "type.raw").load_txt(ndmin=1).size
            <= limit
        ]
        if len(retained) != len(numpy_indices):
            log.warning(
                "Remove %d systems with more than %d atoms",
                len(numpy_indices) - len(retained),
                limit,
            )
        numpy_indices = retained
        numpy_batch_size = f"max:{limit}"
    elif isinstance(numpy_batch_size, list):
        numpy_batch_size = [numpy_batch_size[index] for index in numpy_indices]
    sources = []
    if numpy_indices:
        sources.append(
            DataSourceConfig(
                "deepmd",
                tuple(group.systems[index] for index in numpy_indices),
                numpy_batch_size,
            )
        )
    for index, path in enumerate(group.systems):
        if is_lmdb(path):
            batch_size = group.batch_size
            if isinstance(batch_size, list):
                batch_size = batch_size[index]
            sources.append(DataSourceConfig("lmdb", (path,), batch_size))
    if not sources:
        raise ValueError(
            f"No system left in data group after applying {group.batch_size!r}."
        )
    return sources


def resolve_data_groups(
    config: DataConfig,
    *,
    formats: tuple[DatasetFormat, ...] = ("deepmd", "lmdb"),
) -> list[DataGroupConfig]:
    """Resolve every group's paths and validate format-dependent options."""
    result = []
    for index, group in enumerate(data_group_configs(config)):
        roots = (
            [group["systems"]]
            if isinstance(group["systems"], str)
            else group["systems"]
        )
        if "lmdb" not in formats and any(is_lmdb(path) for path in roots):
            raise ValueError("LMDB training data requires the pt or pt_expt backend.")
        systems = process_systems(
            group["systems"],
            group.get("rglob_patterns"),
            formats=formats,
            fmt=group.get("format"),
            out_fmt=group.get("out_format", group.get("output_format")),
        )
        if not systems:
            raise ValueError(f"Data group {index} contains no supported data systems.")
        if len(systems) > 1 and any(is_lmdb(path) for path in systems):
            if group.get("auto_prob", "prob_sys_size") not in (
                "prob_sys_size",
                "prob_uniform",
            ):
                raise ValueError(
                    f"Data group {index} expands to multiple sources including LMDB, "
                    "so auto_prob must be prob_sys_size or prob_uniform. "
                    "Use separate training_data groups "
                    "to configure each LMDB's internal system probabilities."
                )
            if group.get("sys_probs") is not None:
                raise ValueError(
                    f"Data group {index} expands to multiple sources including LMDB, "
                    "so sys_probs is not supported. Use training_data groups with prob "
                    "to assign source probabilities."
                )
        batch_size = group.get("batch_size", "auto")
        if isinstance(batch_size, list) and len(batch_size) != len(systems):
            raise ValueError(
                f"Data group {index} has {len(systems)} expanded systems but "
                f"{len(batch_size)} batch_size entries."
            )
        result.append(
            DataGroupConfig(
                systems=tuple(systems),
                batch_size=batch_size,
                auto_prob=group.get("auto_prob"),
                sys_probs=group.get("sys_probs"),
                prob=group.get("prob"),
            )
        )
    return result
