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
from deepmd.dpmodel.utils.dist_check import (
    pair_margin_frame_mask,
    select_frames,
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


def _get_stat_pass_length(data: Any, sys_idx: int) -> int:
    """Return the batch count of one pass over a statistical system."""
    get_stat_numb_batches = getattr(data, "get_stat_numb_batches", None)
    if get_stat_numb_batches is not None:
        return int(get_stat_numb_batches(sys_idx))
    return int(data.get_nbatches()[sys_idx])


def _collect_stat_batches(
    data: Any,
    sys_idx: int,
    nbatches: int,
    min_pair_dist: float,
) -> list[dict[str, Any]]:
    """Draw the normalized batches of one statistical system.

    Without a pair-clearance filter these are the next ``nbatches`` batches
    of the system. With one, the frames that hold a pair inside its window are
    dropped and a batch left empty is replaced by the next one, until ``nbatches``
    batches are kept or the scan ends. The scan covers at least one pass over
    the system, and the system contributes whenever that pass draws a valid
    frame. A pass serves every frame of an LMDB system; the NPY reader serves
    whole batches and reshuffles a set before serving it again, so its pass
    leaves out the frames of a set that do not fill a last batch.

    Parameters
    ----------
    data
        The data source, see :func:`make_stat_input`.
    sys_idx : int
        Index of the statistical system.
    nbatches : int
        Number of batches to keep.
    min_pair_dist : float
        Filter radius the requirement was registered with. A non-positive
        value disables the filter.

    Returns
    -------
    list[dict[str, Any]]
        Normalized batches, each with at least one frame; empty when the
        system holds no batch or no valid frame.
    """
    target = _get_stat_numb_batches(data, sys_idx, nbatches)
    scan_limit = target
    if min_pair_dist > 0.0:
        scan_limit = max(target, _get_stat_pass_length(data, sys_idx))
    batches: list[dict[str, Any]] = []
    for _ in range(scan_limit):
        if len(batches) == target:
            break
        stat_data = _get_stat_batch(data, sys_idx)
        if "natoms_vec" in stat_data:
            stat_data["natoms_vec"] = stat_data["natoms_vec"].astype(np.int32)
        batch = normalize_batch(stat_data)
        frame_mask = pair_margin_frame_mask(batch, min_pair_dist > 0.0)
        if frame_mask is not None and not frame_mask.all():
            if not frame_mask.any():
                continue
            batch = select_frames(batch, frame_mask)
        batches.append(batch)
    return batches


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

    This is a low-level helper used by the TF backend.

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
        for dd in sys_stat:
            if merge_sys:
                for bb in sys_stat[dd]:
                    all_stat[dd].append(bb)
            else:
                all_stat[dd].append(sys_stat[dd])
    return all_stat


def make_stat_input(
    data: Any,
    nbatches: int,
    min_pair_dist: float = 0.0,
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
    min_pair_dist : float
        Radius the training-frame filter was registered with. When
        positive, frames holding a pair inside its window are dropped while
        the batches are drawn, a batch left empty is replaced by the next one
        of its system, and a system without any valid frame is skipped, so the
        statistics see the frames the optimizer sees.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Per-system dicts with concatenated numpy arrays.
    """
    nsystems = _get_stat_nsystems(data)
    log.info(f"Packing data for statistics from {nsystems} systems")

    lst: list[dict[str, np.ndarray]] = []
    for ii in range(nsystems):
        batches = _collect_stat_batches(data, ii, nbatches, min_pair_dist)
        if not batches:
            reason = (
                "no frame keeps every atom pair beyond the filter radius"
                if min_pair_dist > 0.0
                else "it holds no batch"
            )
            log.info(f"Skipping statistical system {ii}: {reason}.")
            continue
        # Arrays of two or more dimensions lead with the frame axis; every
        # other entry (find_* flags, an absent box) is constant per system.
        lst.append(
            {
                key: np.concatenate([batch[key] for batch in batches], axis=0)
                if isinstance(value, np.ndarray) and value.ndim >= 2
                else value
                for key, value in batches[0].items()
            }
        )
    if min_pair_dist > 0.0 and not lst:
        raise RuntimeError(
            "No sampled frame keeps every atom pair beyond the filter radius; "
            "lower that radius or clean the dataset."
        )
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
