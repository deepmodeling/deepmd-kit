# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os

import numpy as np
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


def compute_output_bias(energy, natoms, rcond=None):
    """Update output bias for fitting net.

    Args:
    - energy: Batched energy with shape [nframes, 1].
    - natoms: Batched atom statisics with shape [self.ntypes+2].

    Returns
    -------
    - energy_coef: Average enery per atom for each element.
    """
    for i in range(len(energy)):
        energy[i] = energy[i].mean(dim=0, keepdim=True)
        natoms[i] = natoms[i].double().mean(dim=0, keepdim=True)
    sys_ener = torch.cat(energy).cpu()
    sys_tynatom = torch.cat(natoms)[:, 2:].cpu()
    energy_coef, _, _, _ = np.linalg.lstsq(sys_tynatom, sys_ener, rcond)
    return energy_coef


def process_stat_path(
    stat_file_dict, stat_file_dir, model_params_dict, descriptor_cls, fitting_cls
):
    if stat_file_dict is None:
        stat_file_dict = {}
        if "descriptor" in model_params_dict:
            default_stat_file_name_descrpt = descriptor_cls.get_stat_name(
                len(model_params_dict["type_map"]),
                model_params_dict["descriptor"]["type"],
                **model_params_dict["descriptor"],
            )
            stat_file_dict["descriptor"] = default_stat_file_name_descrpt
        if "fitting_net" in model_params_dict:
            default_stat_file_name_fitting = fitting_cls.get_stat_name(
                len(model_params_dict["type_map"]),
                model_params_dict["fitting_net"].get("type", "ener"),
                **model_params_dict["fitting_net"],
            )
            stat_file_dict["fitting_net"] = default_stat_file_name_fitting
    stat_file_path = {
        key: os.path.join(stat_file_dir, stat_file_dict[key]) for key in stat_file_dict
    }

    has_stat_file_path_list = [
        os.path.exists(stat_file_path[key]) for key in stat_file_dict
    ]
    return stat_file_path, False not in has_stat_file_path_list
