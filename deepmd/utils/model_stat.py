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


def _make_all_stat_ref(data: Any, nbatches: int) -> dict[str, list[Any]]:
    all_stat = defaultdict(list)
    for ii in range(data.get_nsystems()):
        for jj in range(nbatches):
            stat_data = data.get_batch(sys_idx=ii)
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
        The data (must support ``get_nsystems()`` and ``get_batch(sys_idx=)``)
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
    for ii in range(data.get_nsystems()):
        sys_stat = defaultdict(list)
        for jj in range(nbatches):
            stat_data = data.get_batch(sys_idx=ii)
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
) -> list[dict[str, np.ndarray]]:
    """Pack data for statistics using DeepmdDataSystem.

    Collects *nbatches* batches from each system and concatenates them
    into a single dict per system.  The returned format
    (``list[dict[str, np.ndarray]]``) is backend-agnostic and can be
    consumed by ``compute_or_load_stat`` in dpmodel, pt_expt, and jax.

    Parameters
    ----------
    data
        The multi-system data manager
        (must support ``get_nsystems()`` and ``get_batch(sys_idx=)``).
    nbatches : int
        Number of batches to collect per system.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Per-system dicts with concatenated numpy arrays.
    """
    all_stat = collect_batches(data, nbatches, merge_sys=False)

    nsystems = data.get_nsystems()
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
