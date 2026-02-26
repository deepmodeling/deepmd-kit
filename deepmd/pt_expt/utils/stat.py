# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import numpy as np

from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.model_stat import make_stat_input as _make_stat_input_raw

log = logging.getLogger(__name__)


def make_stat_input(
    data: DeepmdDataSystem,
    nbatches: int,
) -> list[dict[str, np.ndarray]]:
    """Pack data for statistics using DeepmdDataSystem.

    Collects *nbatches* batches from each system and concatenates them
    into a single dict per system.  The returned format matches the
    ``list[dict[str, np.ndarray]]`` expected by
    ``compute_or_load_stat``.

    Parameters
    ----------
    data : DeepmdDataSystem
        The multi-system data manager.
    nbatches : int
        Number of batches to collect per system.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Per-system dicts with concatenated numpy arrays.
    """
    # Reuse the shared helper with merge_sys=False so that data is
    # grouped by system:  all_stat[key][sys_idx] = [batch0, batch1, ...]
    all_stat = _make_stat_input_raw(data, nbatches, merge_sys=False)

    nsystems = data.get_nsystems()
    log.info(f"Packing data for statistics from {nsystems} systems")

    # Transpose dict-of-lists-of-lists → list-of-dicts and concatenate
    # batches within each system.
    keys = list(all_stat.keys())
    lst: list[dict[str, np.ndarray]] = []
    for ii in range(nsystems):
        merged: dict[str, np.ndarray] = {}
        for key in keys:
            vals = all_stat[key][ii]  # list of batch arrays for this system
            if isinstance(vals[0], np.ndarray):
                if vals[0].ndim >= 2:
                    # 2D+ arrays (e.g. coord [nf, natoms*3]) — concat along axis 0
                    merged[key] = np.concatenate(vals, axis=0)
                else:
                    # 1D arrays (e.g. natoms_vec [2+ntypes]) — per-system
                    # constant, just keep one copy
                    merged[key] = vals[0]
            else:
                # scalar flags like find_*
                merged[key] = vals[0]

        # DeepmdDataSystem.get_batch() uses "type" but the stat system
        # (env_mat_stat, compute_output_stats, etc.) expects "atype".
        if "type" in merged and "atype" not in merged:
            merged["atype"] = merged.pop("type")

        # Reshape coord from [nf, natoms*3] → [nf, natoms, 3]
        if "atype" in merged and "coord" in merged:
            natoms = merged["atype"].shape[-1]
            merged["coord"] = merged["coord"].reshape(-1, natoms, 3)

        # Provide "natoms" from "natoms_vec" (expected by stat system).
        # natoms_vec from get_batch() is 1D [2+ntypes], but
        # compute_output_stats expects 2D [nframes, 2+ntypes].
        if "natoms_vec" in merged and "natoms" not in merged:
            nv = merged["natoms_vec"]
            if nv.ndim == 1:
                nframes = merged["coord"].shape[0]
                nv = np.tile(nv, (nframes, 1))
            merged["natoms"] = nv

        lst.append(merged)
    return lst
