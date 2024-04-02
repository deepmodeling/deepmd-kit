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
    env,
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
    compute_stats_from_atomic,
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


def restore_from_file(
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


def save_to_file(
    stat_file_path: DPPath,
    results: dict,
):
    assert stat_file_path is not None
    stat_file_path.mkdir(exist_ok=True, parents=True)
    for kk, vv in results.items():
        fp = stat_file_path / f"bias_atom_{kk}"
        fp.save_numpy(vv)


def compute_output_stats(
    merged: Union[Callable[[], List[dict]], List[dict]],
    ntypes: int,
    keys: List[str],
    stat_file_path: Optional[DPPath] = None,
    rcond: Optional[float] = None,
    atom_ener: Optional[List[float]] = None,
    model_forward: Optional[Callable[..., torch.Tensor]] = None,
):
    key_mapping = {
        "polar": "polarizability",
        "energy": "energy",
        "dos": "dos",
        "dipole": "dipole",
    }

    if (
        "energy" in keys
    ):  # this is the energy fitting which may have keys ['energy', 'dforce']
        return compute_output_stats_global_only(
            merged=merged,
            ntypes=ntypes,
            keys=[key_mapping[k] for k in keys],
            stat_file_path=stat_file_path,
            rcond=rcond,
            atom_ener=atom_ener,
            model_forward=model_forward,
        )
    elif (
        "dos" in keys or "polar" in keys
    ):  # this is the polar fitting or dos fitting which may have keys ['polar'] or ['dos']
        return compute_output_stats_with_atomic(
            merged=merged,
            ntypes=ntypes,
            keys=[key_mapping[k] for k in keys],
            stat_file_path=stat_file_path,
            rcond=rcond,
            atom_ener=atom_ener,
            model_forward=model_forward,
        )
    else:
        # can add mode facade services.
        pass


def compute_output_stats_global_only(
    merged: Union[Callable[[], List[dict]], List[dict]],
    ntypes: int,
    keys: List[str],
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
    keys : List[int]
        The output variable names of a given atomic model, can be found in `fitting_output_def` of the model.
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
    bias_atom_e = restore_from_file(stat_file_path, keys)
    if bias_atom_e is None:
        if callable(merged):
            # only get data for once
            sampled = merged()
        else:
            sampled = merged
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
            # only use statistics result
            # [0]: take the first otuput (mean) of compute_stats_from_redu
            bias_atom_e = {
                kk: compute_stats_from_redu(
                    merged_output[kk],
                    merged_natoms,
                    assigned_bias=assigned_atom_ener,
                    rcond=rcond,
                )[0]
                for kk in keys
            }
        else:
            # subtract the model bias and output the delta bias
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

            bias_diff = {kk: merged_output[kk] - model_predict[kk] for kk in keys}
            bias_atom_e = {
                kk: compute_stats_from_redu(
                    bias_diff[kk],
                    merged_natoms,
                    assigned_bias=assigned_atom_ener,
                    rcond=rcond,
                )[0]
                for kk in keys
            }
            unbias_e = {
                kk: model_predict[kk] + merged_natoms @ bias_atom_e[kk] for kk in keys
            }
            atom_numbs = merged_natoms.sum(-1)
            for kk in keys:
                rmse_ae = np.sqrt(
                    np.mean(
                        np.square(
                            (unbias_e[kk].ravel() - merged_output[kk].ravel())
                            / atom_numbs
                        )
                    )
                )
                log.info(
                    f"RMSE of {kk} per atom after linear regression is: {rmse_ae} in the unit of {kk}."
                )

        if stat_file_path is not None:
            save_to_file(stat_file_path, bias_atom_e)

    ret = {kk: to_torch_tensor(bias_atom_e[kk]) for kk in keys}

    return ret


def compute_output_stats_with_atomic(
    merged: Union[Callable[[], List[dict]], List[dict]],
    ntypes: int,
    keys: List[str],
    stat_file_path: Optional[DPPath] = None,
    rcond: Optional[float] = None,
    atom_ener: Optional[List[float]] = None,
    model_forward: Optional[Callable[..., torch.Tensor]] = None,
):
    """
    Compute the output statistics (e.g. dos bias) for the fitting net from packed data.

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
    keys : str
        The var_name of the fitting net.
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
    atom_bias = restore_from_file(stat_file_path, keys)

    if atom_bias is None:
        if callable(merged):
            # only get data for once
            sampled = merged()
        else:
            sampled = merged
        if model_forward is not None:
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

                # need to check outputs
                sample_predict = model_forward_auto_batch_size(
                    coord, atype, box, fparam=fparam, aparam=aparam
                )
                for kk in keys:
                    model_predict[kk].append(
                        to_numpy_array(
                            sample_predict[kk]  # nf x nloc x odims
                        )
                    )

            # this stores all atomic predictions.
            model_predict = {kk: np.concatenate(model_predict[kk]) for kk in keys}

        else:
            model_predict = {}

        atom_bias = {kk: [] for kk in keys}
        for idx, system in enumerate(sampled):
            nframs = system["atype"].shape[0]

            for kk in keys:
                if system["find_atom_" + kk] > 0.0:
                    sys_property = system["atom_" + kk].numpy(force=True)
                    if kk not in model_predict:
                        sys_bias = compute_stats_from_atomic(
                            sys_property,
                            system["atype"].numpy(force=True),
                        )[0]
                    else:
                        bias_diff = sys_property - model_predict[kk][idx]
                        sys_bias = compute_stats_from_atomic(
                            bias_diff,
                            system["atype"].numpy(force=True),
                        )[0]
                else:
                    if not system["find_" + kk] > 0.0:
                        continue
                    sys_type_count = np.zeros(
                        (nframs, ntypes), dtype=env.GLOBAL_NP_FLOAT_PRECISION
                    )
                    for itype in range(ntypes):
                        type_mask = system["atype"] == itype
                        sys_type_count[:, itype] = type_mask.sum(dim=1).numpy(
                            force=True
                        )
                    sys_bias_redu = system[kk].numpy(force=True)
                    if kk not in model_predict:
                        sys_bias = compute_stats_from_redu(
                            sys_bias_redu, sys_type_count, rcond=rcond
                        )[0]
                    else:
                        bias_diff = sys_bias_redu - model_predict[kk][idx].sum(-1)
                        sys_bias = compute_stats_from_redu(
                            bias_diff, sys_type_count, rcond=rcond
                        )[0]

            atom_bias[kk].append(sys_bias)
            # need to take care shift diag and add atom_ener
        atom_bias = {
            kk: np.nan_to_num(np.stack(vv).mean(axis=0)) for kk, vv in atom_bias.items()
        }
        if stat_file_path is not None:
            save_to_file(stat_file_path, atom_bias)
    ret = {kk: to_torch_tensor(atom_bias[kk]) for kk in keys}
    return ret
