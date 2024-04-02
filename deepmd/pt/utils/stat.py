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
        return None
    stat_files = [stat_file_path / f"bias_atom_{kk}" for kk in keys]
    if any(not (ii.is_file()) for ii in stat_files):
        return None
    ret = {}

    for kk in keys:
        fp = stat_file_path / f"bias_atom_{kk}"
        assert fp.is_file()
        ret[kk] = fp.load_numpy()
    return ret


def _save_to_file(
    stat_file_path: DPPath,
    results: dict,
):
    assert stat_file_path is not None
    stat_file_path.mkdir(exist_ok=True, parents=True)
    for kk, vv in results.items():
        fp = stat_file_path / f"bias_atom_{kk}"
        fp.save_numpy(vv)


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
                    torch.sum(sample_predict[kk], dim=1)  # nf x nloc x odims
                )
            )
    model_predict = {kk: np.concatenate(model_predict[kk]) for kk in keys}
    return model_predict


def compute_output_stats(
    merged: Union[Callable[[], List[dict]], List[dict]],
    ntypes: int,
    keys: Union[str, List[str]] = ["energy"],
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
    # only get data for once
    sampled = merged() if callable(merged) else merged
    # remove the keys that are not in the sample
    new_keys = []
    keys = [keys] if isinstance(keys, str) else keys
    assert isinstance(keys, list)
    for ii in keys:
        if ii in sampled[0].keys():
            new_keys.append(ii)
    del keys
    keys = new_keys

    # try to restore the bias from stat file
    bias_atom_e = _restore_from_file(stat_file_path, keys)

    # failed to restore the bias from stat file. compute
    if bias_atom_e is None:
        outputs = {kk: [item[kk] for item in sampled] for kk in keys}
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
        if atom_ener is not None and len(atom_ener) > 0:
            assigned_atom_ener = np.array(
                [ee if ee is not None else np.nan for ee in atom_ener]
            )
        else:
            assigned_atom_ener = None

        if model_forward is None:
            stats_input = merged_output
        else:
            # subtract the model bias and output the delta bias
            model_predict = _compute_model_predict(sampled, keys, model_forward)
            stats_input = {kk: merged_output[kk] - model_predict[kk] for kk in keys}

        # [0]: take the first otuput (mean) of compute_stats_from_redu
        bias_atom_e = {
            kk: compute_stats_from_redu(
                stats_input[kk],
                merged_natoms,
                assigned_bias=assigned_atom_ener,
                rcond=rcond,
            )[0]
            for kk in keys
        }

        if model_forward is None:
            unbias_e = {kk: merged_natoms @ bias_atom_e[kk] for kk in keys}
        else:
            unbias_e = {
                kk: model_predict[kk] + merged_natoms @ bias_atom_e[kk] for kk in keys
            }
        atom_numbs = merged_natoms.sum(-1)
        for kk in keys:
            rmse_ae = np.sqrt(
                np.mean(
                    np.square(
                        (unbias_e[kk].ravel() - merged_output[kk].ravel()) / atom_numbs
                    )
                )
            )
            log.info(
                f"RMSE of {kk} per atom after linear regression is: {rmse_ae} in the unit of {kk}."
            )

        if stat_file_path is not None:
            _save_to_file(stat_file_path, bias_atom_e)

    ret = {kk: to_torch_tensor(bias_atom_e[kk]) for kk in keys}

    return ret
