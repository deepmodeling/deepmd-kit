# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.pt.utils import (
    AtomExcludeMask,
)
from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pt.utils.utils import (
    dict_to_device,
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)
from deepmd.utils.path import (
    DPPath,
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


def compute_output_stats(
    merged: Union[Callable[[], List[dict]], List[dict]],
    ntypes: int,
    stat_file_path: Optional[DPPath] = None,
    rcond: Optional[float] = None,
    atom_ener: Optional[List[float]] = None,
    model_forward: Optional[Callable[..., torch.Tensor]] = None,
):
    """
    Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

    Parameters
    ----------
    merged : Union[Callable[[], List[dict]], List[dict]]
        - List[dict]: A list of data samples from various data systems.
            Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
            originating from the `i`-th data system.
        - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
            only when needed. Since the sampling process can be slow and memory-intensive,
            the lazy function helps by only sampling once.
    ntypes : int
        The number of atom types.
    stat_file_path : DPPath, optional
        The path to the stat file.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    atom_ener : List[float], optional
        Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.
    model_forward : Callable[..., torch.Tensor], optional
        The wrapped forward function of atomic model.
        If not None, the model will be utilized to generate the original energy prediction,
        which will be subtracted from the energy label of the data.
        The difference will then be used to calculate the delta complement energy bias for each type.
    """
    if stat_file_path is not None:
        stat_file_path = stat_file_path / "bias_atom_e"
    if stat_file_path is not None and stat_file_path.is_file():
        bias_atom_e = stat_file_path.load_numpy()
    else:
        if callable(merged):
            # only get data for once
            sampled = merged()
        else:
            sampled = merged
        energy = [item["energy"] for item in sampled]
        data_mixed_type = "real_natoms_vec" in sampled[0]
        natoms_key = "natoms" if not data_mixed_type else "real_natoms_vec"
        for system in sampled:
            if "atom_exclude_types" in system:
                type_mask = AtomExcludeMask(
                    ntypes, system["atom_exclude_types"]
                ).get_type_mask()
                system[natoms_key][:, 2:] *= type_mask.unsqueeze(0)
        input_natoms = [item[natoms_key] for item in sampled]
        # shape: (nframes, ndim)
        merged_energy = to_numpy_array(torch.cat(energy))
        # shape: (nframes, ntypes)
        merged_natoms = to_numpy_array(torch.cat(input_natoms)[:, 2:])
        if atom_ener is not None and len(atom_ener) > 0:
            assigned_atom_ener = np.array(
                [ee if ee is not None else np.nan for ee in atom_ener]
            )
        else:
            assigned_atom_ener = None
        if model_forward is None:
            # only use statistics result
            bias_atom_e, _ = compute_stats_from_redu(
                merged_energy,
                merged_natoms,
                assigned_bias=assigned_atom_ener,
                rcond=rcond,
            )
        else:
            # subtract the model bias and output the delta bias
            auto_batch_size = AutoBatchSize()
            energy_predict = []
            for system in sampled:
                nframes = system["coord"].shape[0]
                coord, atype, box, natoms = (
                    system["coord"],
                    system["atype"],
                    system["box"],
                    system["natoms"],
                )
                fparam = system["fparam"] if "fparam" in system else None
                aparam = system["aparam"] if "aparam" in system else None

                def model_forward_auto_batch_size(*args, **kwargs):
                    return auto_batch_size.execute_all(
                        model_forward,
                        nframes,
                        system["atype"].shape[-1],
                        *args,
                        **kwargs,
                    )

                energy = (
                    model_forward_auto_batch_size(
                        coord, atype, box, fparam=fparam, aparam=aparam
                    )
                    .reshape(nframes, -1)
                    .sum(-1)
                )
                energy_predict.append(to_numpy_array(energy).reshape([nframes, 1]))

            energy_predict = np.concatenate(energy_predict)
            bias_diff = merged_energy - energy_predict
            bias_atom_e, _ = compute_stats_from_redu(
                bias_diff,
                merged_natoms,
                assigned_bias=assigned_atom_ener,
                rcond=rcond,
            )
            unbias_e = energy_predict + merged_natoms @ bias_atom_e
            atom_numbs = merged_natoms.sum(-1)
            rmse_ae = np.sqrt(
                np.mean(
                    np.square((unbias_e.ravel() - merged_energy.ravel()) / atom_numbs)
                )
            )
            log.info(
                f"RMSE of energy per atom after linear regression is: {rmse_ae} eV/atom."
            )
        if stat_file_path is not None:
            stat_file_path.save_numpy(bias_atom_e)
    assert all(x is not None for x in [bias_atom_e])
    return to_torch_tensor(bias_atom_e)
