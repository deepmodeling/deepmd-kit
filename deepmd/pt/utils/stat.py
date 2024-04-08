# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Callable,
    Dict,
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
                    elif isinstance(stat_data[dd], np.float32):
                        sys_stat[dd] = stat_data[dd]
                    else:
                        pass

        for key in sys_stat:
            if isinstance(sys_stat[key], np.float32):
                pass
            elif sys_stat[key] is None or sys_stat[key][0] is None:
                sys_stat[key] = None
            elif isinstance(stat_data[dd], torch.Tensor):
                sys_stat[key] = torch.cat(sys_stat[key], dim=0)
        dict_to_device(sys_stat)
        lst.append(sys_stat)
    return lst


def _restore_from_file(
    stat_file_path: DPPath,
    keys: List[str] = ["energy"],
) -> Optional[dict]:
    if stat_file_path is None:
        return None, None
    stat_files = [stat_file_path / f"bias_atom_{kk}" for kk in keys]
    if all(not (ii.is_file()) for ii in stat_files):
        return None, None
    stat_files = [stat_file_path / f"std_atom_{kk}" for kk in keys]
    if all(not (ii.is_file()) for ii in stat_files):
        return None, None

    ret_bias = {}
    ret_std = {}
    for kk in keys:
        fp = stat_file_path / f"bias_atom_{kk}"
        # only read the key that exists
        if fp.is_file():
            ret_bias[kk] = fp.load_numpy()
    for kk in keys:
        fp = stat_file_path / f"std_atom_{kk}"
        # only read the key that exists
        if fp.is_file():
            ret_std[kk] = fp.load_numpy()
    return ret_bias, ret_std


def _save_to_file(
    stat_file_path: DPPath,
    bias_out: dict,
    std_out: dict,
):
    assert stat_file_path is not None
    stat_file_path.mkdir(exist_ok=True, parents=True)
    for kk, vv in bias_out.items():
        fp = stat_file_path / f"bias_atom_{kk}"
        fp.save_numpy(vv)
    for kk, vv in std_out.items():
        fp = stat_file_path / f"std_atom_{kk}"
        fp.save_numpy(vv)


def _post_process_stat(
    out_bias,
    out_std,
):
    """Post process the statistics.

    For global statistics, we do not have the std for each type of atoms,
    thus fake the output std by ones for all the types.

    """
    new_std = {}
    for kk, vv in out_bias.items():
        new_std[kk] = np.ones_like(vv)
    return out_bias, new_std


def _compute_model_predict(
    sampled: Union[Callable[[], List[dict]], List[dict]],
    keys: List[str],
    model_forward: Callable[..., torch.Tensor],
):
    auto_batch_size = AutoBatchSize()
    model_predict = {kk: [] for kk in keys}
    for system in sampled:
        nframes = system["coord"].shape[0]
        coord, atype, box, natoms = (
            system["coord"],
            system["atype"],
            system["box"],
            system["natoms"],
        )
        fparam = system.get("fparam", None)
        aparam = system.get("aparam", None)

        def model_forward_auto_batch_size(*args, **kwargs):
            return auto_batch_size.execute_all(
                model_forward,
                nframes,
                system["atype"].shape[-1],
                *args,
                **kwargs,
            )

        sample_predict = model_forward_auto_batch_size(
            coord, atype, box, fparam=fparam, aparam=aparam
        )
        for kk in keys:
            model_predict[kk].append(
                to_numpy_array(
                    sample_predict[kk]  # nf x nloc x odims
                )
            )
    model_predict = {kk: np.concatenate(model_predict[kk]) for kk in keys}
    return model_predict


def _make_preset_out_bias(
    ntypes: int,
    ibias: List[Optional[np.array]],
) -> Optional[np.array]:
    """Make preset out bias.

    output:
        a np array of shape [ntypes, *(odim0, odim1, ...)] is any item is not None
        None if all items are None.
    """
    if len(ibias) != ntypes:
        raise ValueError("the length of preset bias list should be ntypes")
    if all(ii is None for ii in ibias):
        return None
    for refb in ibias:
        if refb is not None:
            break
    refb = np.array(refb)
    nbias = [
        np.full_like(refb, np.nan, dtype=np.float64) if ii is None else ii
        for ii in ibias
    ]
    return np.array(nbias)


def compute_output_stats(
    merged: Union[Callable[[], List[dict]], List[dict]],
    ntypes: int,
    keys: Union[str, List[str]] = ["energy"],
    stat_file_path: Optional[DPPath] = None,
    rcond: Optional[float] = None,
    preset_bias: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
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
    preset_bias : Dict[str, List[Optional[torch.Tensor]]], optional
        Specifying atomic energy contribution in vacuum. Given by key:value pairs.
        The value is a list specifying the bias. the elements can be None or np.array of output shape.
        For example: [None, [2.]] means type 0 is not set, type 1 is set to [2.]
        The `set_davg_zero` key in the descrptor should be set.
    model_forward : Callable[..., torch.Tensor], optional
        The wrapped forward function of atomic model.
        If not None, the model will be utilized to generate the original energy prediction,
        which will be subtracted from the energy label of the data.
        The difference will then be used to calculate the delta complement energy bias for each type.
    """
    # try to restore the bias from stat file
    bias_atom_e, std_atom_e = _restore_from_file(stat_file_path, keys)

    # failed to restore the bias from stat file. compute
    if bias_atom_e is None:
        
        
        # only get data once, sampled is a list of dict[str, torch.Tensor]
        sampled = merged() if callable(merged) else merged 
        if model_forward is not None:
            model_pred = _compute_model_predict(sampled, keys, model_forward)
        else:
            model_pred = None

        # split system based on label
        atomic_sampled = {}
        global_sampled = {}
        """
        case1: system-1 global dipole and atomic polar, system-2 global dipole and global polar
            dipole,sys1 --> add to global_sampled
            dipole,sys2 --> add to global_sampled
            polar, sys1 --> add to atomic_sampled
            polar, sys2 --> do nothing
            global_sampled : [sys1, sys2]
            atomic_sampled : [sys1]
        """
        for kk in  keys:
            for idx, system in enumerate(sampled):
                if (("find_atom_" + kk) in system) and (system["find_atom_" + kk] > 0.0) and (idx not in atomic_sampled):
                   atomic_sampled[idx] = system
                elif (("find_" + kk) in system) and (system["find_" + kk] > 0.0) and (idx not in global_sampled):
                    global_sampled[idx] = system
                else:
                    continue

        atomic_sampled = list(atomic_sampled.values())
        global_sampled = list(global_sampled.values())
        if len(global_sampled) > 0:
            bias_atom_e, std_atom_e = compute_output_stats_global(
                global_sampled,
                ntypes,
                keys,
                rcond,
                preset_bias,
                model_pred,
            )
        
        if len(atomic_sampled) > 0:
            bias_atom_e, std_atom_e = compute_output_stats_atomic(
                global_sampled,
                ntypes,
                keys,
                rcond,
                preset_bias,
                model_pred,
            )
        
        # need to merge dict
        if stat_file_path is not None:
            _save_to_file(stat_file_path, bias_atom_e, std_atom_e)

    bias_atom_e = {kk: to_torch_tensor(vv) for kk, vv in bias_atom_e.items()}
    std_atom_e = {kk: to_torch_tensor(vv) for kk, vv in std_atom_e.items()}
    return  bias_atom_e, std_atom_e

def compute_output_stats_global(
    sampled: List[dict],
    ntypes: int,
    keys: List[str],
    rcond: Optional[float] = None,
    preset_bias: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
    model_pred: Optional[Dict[str, np.ndarray]] = None,
):
    """This function only handle stat computation from reduced global labels."""
    
    # remove the keys that are not in the sample
    keys = [keys] if isinstance(keys, str) else keys
    assert isinstance(keys, list)
    new_keys = [ii for ii in keys if ii in sampled[0].keys()]
    del keys
    keys = new_keys

    # get label dict from sample; for each key, only picking the system with global labels.
    outputs = {kk: [system[kk] for system in sampled if kk in system and system.get(f"find_{kk}", 0) > 0] for kk in keys}

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
    merged_output = {kk: to_numpy_array(torch.cat(outputs[kk])) for kk in keys}
    # shape: (nframes, ntypes)
    merged_natoms = to_numpy_array(torch.cat(input_natoms)[:, 2:])
    nf = merged_natoms.shape[0]
    if preset_bias is not None:
        assigned_atom_ener = {
            kk: _make_preset_out_bias(ntypes, preset_bias[kk])
            if kk in preset_bias.keys()
            else None
            for kk in keys
        }
    else:
        assigned_atom_ener = {kk: None for kk in keys}

    if model_pred is None:
        stats_input = merged_output
    else:
        # subtract the model bias and output the delta bias
        model_pred = {kk: np.sum(model_pred[kk], axis=1) for kk in keys}
        stats_input = {kk: merged_output[kk] - model_pred[kk] for kk in keys}

    bias_atom_e = {}
    std_atom_e = {}
    for kk in keys:
        bias_atom_e[kk], std_atom_e[kk] = compute_stats_from_redu(
            stats_input[kk],
            merged_natoms,
            assigned_bias=assigned_atom_ener[kk],
            rcond=rcond,
        )
    bias_atom_e, std_atom_e = _post_process_stat(bias_atom_e, std_atom_e)
    
    # unbias_e is only used for print rmse
    if model_pred is None:
        unbias_e = {
            kk: merged_natoms @ bias_atom_e[kk].reshape(ntypes, -1) for kk in keys
        }
    else:
        unbias_e = {
            kk: model_pred[kk].reshape(nf, -1)
            + merged_natoms @ bias_atom_e[kk].reshape(ntypes, -1)
            for kk in keys
        }
    atom_numbs = merged_natoms.sum(-1)

    def rmse(x):
        return np.sqrt(np.mean(np.square(x)))

    for kk in keys:
        rmse_ae = rmse(
            (unbias_e[kk].reshape(nf, -1) - merged_output[kk].reshape(nf, -1))
            / atom_numbs[:, None]
        )
        log.info(
            f"RMSE of {kk} per atom after linear regression is: {rmse_ae} in the unit of {kk}."
        )
    return bias_atom_e, std_atom_e

def compute_output_stats_atomic(
    sampled: List[dict],
    ntypes: int,
    keys: List[str],
    rcond: Optional[float] = None,
    preset_bias: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
    model_pred: Optional[Dict[str, np.ndarray]] = None,
):
    pass