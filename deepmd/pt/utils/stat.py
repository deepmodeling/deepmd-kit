# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import torch

from deepmd.pt.utils.utils import (
    dict_to_device,
)

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
    log.info(f"Packing data for statistics from {len(datasets)} systems")
    for i in range(len(datasets)):
        sys_stat = {}
        with torch.device("cpu"):
            iterator = iter(dataloaders[i])
            for _ in range(nbatches):
                try:
                    stat_data = next(iterator)
                except StopIteration:
                    iterator = iter(dataloaders[i])
                    stat_data = next(iterator)
                for dd in stat_data:
                    if stat_data[dd] is None:
                        sys_stat[dd] = None
                    elif isinstance(stat_data[dd], torch.Tensor):
                        if dd not in sys_stat:
                            sys_stat[dd] = []
                        sys_stat[dd].append(stat_data[dd])
                    else:
                        pass
        for key in sys_stat:
            if sys_stat[key] is None or sys_stat[key][0] is None:
                sys_stat[key] = None
            else:
                sys_stat[key] = torch.cat(sys_stat[key], dim=0)
        dict_to_device(sys_stat)
        lst.append(sys_stat)
    return lst
