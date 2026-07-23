# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections import (
    defaultdict,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.batch import (
    normalize_batch,
)

log = logging.getLogger(__name__)


def _get_stat_nsystems(data: Any) -> int:
    """Return the number of shape-compatible systems used for statistics."""
    get_stat_nsystems = getattr(data, "get_stat_nsystems", None)
    if get_stat_nsystems is not None:
        return int(get_stat_nsystems())
    return int(data.get_nsystems())


def _get_stat_numb_batches(data: Any, sys_idx: int, nbatches: int) -> int:
    """Limit sampling to the available batches of one statistical system."""
    get_stat_numb_batches = getattr(data, "get_stat_numb_batches", None)
    if get_stat_numb_batches is None:
        return nbatches
    return min(nbatches, int(get_stat_numb_batches(sys_idx)))


def _get_stat_batch(data: Any, sys_idx: int) -> dict[str, Any]:
    """Return one batch from a shape-compatible statistical system."""
    get_stat_batch = getattr(data, "get_stat_batch", None)
    if get_stat_batch is not None:
        return get_stat_batch(sys_idx)
    return data.get_batch(sys_idx=sys_idx)


def _make_all_stat_ref(data: Any, nbatches: int) -> dict[str, list[Any]]:
    all_stat = defaultdict(list)
    for ii in range(_get_stat_nsystems(data)):
        for jj in range(_get_stat_numb_batches(data, ii, nbatches)):
            stat_data = _get_stat_batch(data, ii)
            for dd in stat_data:
                if dd == "natoms_vec":
                    stat_data[dd] = stat_data[dd].astype(np.int32)
                all_stat[dd].append(stat_data[dd])
    return all_stat


def collect_batches(
    data: Any, nbatches: int, merge_sys: bool = True
) -> dict[str, list[Any]]:
    """Collect batches from a DeepmdDataSystem into a dict of lists.

    This is a low-level helper used by the TF backend and by
    :func:`make_stat_input`.

    Parameters
    ----------
    data
        The data source. It must support ``get_nsystems()`` and
        ``get_batch(sys_idx=)``. Optional ``get_stat_nsystems()``,
        ``get_stat_numb_batches(sys_idx)``, and ``get_stat_batch(sys_idx)``
        hooks may expose shape-compatible logical systems and their available
        batches specifically for statistics.
    nbatches : int
        The number of batches per system
    merge_sys : bool (True)
        Merge system data

    Returns
    -------
    all_stat:
        A dictionary of list of list storing data for stat.
        if merge_sys == False data can be accessed by
            all_stat[key][sys_idx][batch_idx][frame_idx]
        else merge_sys == True can be accessed by
            all_stat[key][batch_idx][frame_idx]
    """
    all_stat = defaultdict(list)
    for ii in range(_get_stat_nsystems(data)):
        sys_stat = defaultdict(list)
        for jj in range(_get_stat_numb_batches(data, ii, nbatches)):
            stat_data = _get_stat_batch(data, ii)
            for dd in stat_data:
                if dd == "natoms_vec":
                    stat_data[dd] = stat_data[dd].astype(np.int32)
                sys_stat[dd].append(stat_data[dd])
        _append_missing_type_frames(data, ii, sys_stat)
        for dd in sys_stat:
            if merge_sys:
                for bb in sys_stat[dd]:
                    all_stat[dd].append(bb)
            else:
                all_stat[dd].append(sys_stat[dd])
    return all_stat


def _append_missing_type_frames(
    data: Any, sys_idx: int, sys_stat: dict[str, list[Any]]
) -> None:
    """Append representative mixed-type frames for types missed by sampling.

    Energy/output bias statistics regress one bias per atom type from the sampled
    frame compositions.  Mixed-type systems can contain types that do not appear
    in the small random statistics sample.  When that happens, append the first
    frame containing each missing type so the regression is constrained for every
    type that exists in the underlying system.  Standard (non-mixed) systems have
    fixed composition and do not need augmentation.
    """
    if "real_natoms_vec" not in sys_stat or not hasattr(data, "data_systems"):
        return
    if getattr(data, "mixed_systems", False):
        # In mixed-system batching sys_idx is intentionally ignored by get_batch;
        # keep the historical sampling behaviour rather than guessing ownership.
        return
    data_system = data.data_systems[sys_idx]
    dataset_counts, first_frame_for_type = _mixed_type_coverage(data_system)
    if dataset_counts is None or first_frame_for_type is None:
        return
    sampled_counts = np.concatenate(sys_stat["real_natoms_vec"], axis=0)[:, 2:].sum(
        axis=0
    )
    missing_types = np.flatnonzero((dataset_counts > 0) & (sampled_counts == 0))
    if len(missing_types) == 0:
        return

    used_frames: set[int] = set()
    while len(missing_types) > 0:
        frame_idx: int | None = None
        for type_i in missing_types:
            candidate = int(first_frame_for_type[int(type_i)])
            if candidate >= 0 and candidate not in used_frames:
                frame_idx = candidate
                break
        if frame_idx is None:
            break
        used_frames.add(frame_idx)
        extra_batch = data_system.get_single_frame(frame_idx, num_worker=1)
        extra_batch["natoms_vec"] = data.natoms_vec[sys_idx].astype(np.int32)
        extra_batch["default_mesh"] = data.default_mesh[sys_idx]
        for key, value in extra_batch.items():
            if (
                key not in {"natoms_vec", "default_mesh"}
                and isinstance(value, np.ndarray)
                and value.ndim >= 1
            ):
                value = value.reshape((1, *value.shape))
            sys_stat[key].append(value)
        sampled_counts += (
            extra_batch["real_natoms_vec"].reshape(1, -1)[:, 2:].sum(axis=0)
        )
        missing_types = np.flatnonzero((dataset_counts > 0) & (sampled_counts == 0))


def _mixed_type_coverage(
    data_system: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return full mixed-type counts and a representative frame per type."""
    if not getattr(data_system, "mixed_type", False):
        return None, None
    ntypes = data_system.get_ntypes()
    counts = np.zeros(ntypes, dtype=np.int64)
    first_frame_for_type = np.full(ntypes, -1, dtype=np.int64)
    frame_offset = 0
    for set_dir, frame_end in zip(
        data_system.dirs, data_system.prefix_sum, strict=True
    ):
        real_type = (set_dir / "real_atom_types.npy").load_numpy()
        if getattr(data_system, "enforce_type_map", False):
            real_type = data_system.type_idx_map[real_type].astype(np.int32)
        real_type = real_type.reshape(frame_end - frame_offset, data_system.natoms)
        for type_i in range(ntypes):
            frame_hits = np.flatnonzero((real_type == type_i).any(axis=1))
            counts[type_i] += int((real_type == type_i).sum())
            if first_frame_for_type[type_i] < 0 and len(frame_hits) > 0:
                first_frame_for_type[type_i] = frame_offset + int(frame_hits[0])
        frame_offset = frame_end
    return counts, first_frame_for_type


def make_stat_input(
    data: Any,
    nbatches: int,
) -> list[dict[str, np.ndarray]]:
    """Pack data for statistics using DeepmdDataSystem.

    Collects up to *nbatches* batches from each shape-compatible statistical
    system and concatenates them into one dictionary per system. Data sources
    with variable atom counts may expose dedicated statistical-system methods
    so incompatible ``nloc`` groups remain separate. The returned format
    (``list[dict[str, np.ndarray]]``) is backend-agnostic and can be
    consumed by ``compute_or_load_stat`` in dpmodel, pt_expt, and jax.

    Parameters
    ----------
    data
        The multi-system data manager. It must support ``get_nsystems()`` and
        ``get_batch(sys_idx=)``. Optional ``get_stat_nsystems()``,
        ``get_stat_numb_batches(sys_idx)``, and ``get_stat_batch(sys_idx)``
        hooks may expose shape-compatible logical systems and their available
        batches specifically for statistics.
    nbatches : int
        Number of batches to collect per system.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Per-system dicts with concatenated numpy arrays.
    """
    all_stat = collect_batches(data, nbatches, merge_sys=False)

    nsystems = _get_stat_nsystems(data)
    log.info(f"Packing data for statistics from {nsystems} systems")

    keys = list(all_stat.keys())
    lst: list[dict[str, np.ndarray]] = []
    for ii in range(nsystems):
        merged: dict[str, np.ndarray] = {}
        for key in keys:
            vals = all_stat[key][ii]  # list of batch arrays for this system
            if isinstance(vals[0], np.ndarray):
                if vals[0].ndim >= 2:
                    merged[key] = np.concatenate(vals, axis=0)
                else:
                    # 1D arrays (e.g. natoms_vec) — per-system constant
                    merged[key] = vals[0]
            else:
                # scalar flags like find_*
                merged[key] = vals[0]

        lst.append(normalize_batch(merged))
    return lst


def merge_sys_stat(all_stat: dict[str, list[Any]]) -> dict[str, list[Any]]:
    first_key = next(iter(all_stat.keys()))
    nsys = len(all_stat[first_key])
    ret = defaultdict(list)
    for ii in range(nsys):
        for dd in all_stat:
            for bb in all_stat[dd][ii]:
                ret[dd].append(bb)
    return ret
