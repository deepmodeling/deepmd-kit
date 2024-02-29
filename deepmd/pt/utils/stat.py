# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import torch

log = logging.getLogger(__name__)


def make_stat_input(datasets, dataloaders, nbatches):
    """Pack data for statistics.

    Args:
    - dataset: A list of dataset to analyze.
    - nbatches: Batch count for collecting stats.

    Returns
    -------
    - a list of dicts, each of which contains data from a system
    """
    lst = []
    keys = [
        "coord",
        "force",
        "energy",
        "atype",
        "box",
        "natoms",
    ]
    if datasets[0].mixed_type:
        keys.append("real_natoms_vec")
    log.info(f"Packing data for statistics from {len(datasets)} systems")
    for i in range(len(datasets)):
        sys_stat = {key: [] for key in keys}
        with torch.device("cpu"):
            iterator = iter(dataloaders[i])
            for _ in range(nbatches):
                try:
                    stat_data = next(iterator)
                except StopIteration:
                    iterator = iter(dataloaders[i])
                    stat_data = next(iterator)
                for dd in stat_data:
                    if dd in keys:
                        sys_stat[dd].append(stat_data[dd])
        for key in keys:
            if not isinstance(sys_stat[key][0], list):
                if sys_stat[key][0] is None:
                    sys_stat[key] = None
                else:
                    sys_stat[key] = torch.cat(sys_stat[key], dim=0)
            else:
                sys_stat_list = []
                for ii, _ in enumerate(sys_stat[key][0]):
                    tmp_stat = [x[ii] for x in sys_stat[key]]
                    sys_stat_list.append(torch.cat(tmp_stat, dim=0))
                sys_stat[key] = sys_stat_list
        lst.append(sys_stat)
    return lst
