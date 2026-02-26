# SPDX-License-Identifier: LGPL-3.0-or-later
"""Output statistics computation for dpmodel backend."""

import logging
from collections import (
    defaultdict,
)
from collections.abc import (
    Callable,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.utils.out_stat import (
    compute_stats_do_not_distinguish_types,
    compute_stats_from_atomic,
    compute_stats_from_redu,
)
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)


def _restore_from_file(
    stat_file_path: DPPath,
    keys: list[str],
) -> tuple[dict | None, dict | None]:
    """Restore bias and std from stat file."""
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
        if fp.is_file():
            ret_bias[kk] = fp.load_numpy()
    for kk in keys:
        fp = stat_file_path / f"std_atom_{kk}"
        if fp.is_file():
            ret_std[kk] = fp.load_numpy()
    return ret_bias, ret_std


def _save_to_file(
    stat_file_path: DPPath,
    bias_out: dict,
    std_out: dict,
) -> None:
    """Save bias and std to stat file."""
    assert stat_file_path is not None
    stat_file_path.mkdir(exist_ok=True, parents=True)
    for kk, vv in bias_out.items():
        fp = stat_file_path / f"bias_atom_{kk}"
        fp.save_numpy(vv)
    for kk, vv in std_out.items():
        fp = stat_file_path / f"std_atom_{kk}"
        fp.save_numpy(vv)


def _post_process_stat(
    out_bias: dict,
    out_std: dict,
) -> tuple[dict, dict]:
    """Post process the statistics.

    For global statistics, we do not have the std for each type of atoms,
    thus fake the output std by ones for all the types.
    If the shape of out_std is already the same as out_bias,
    we do not need to do anything.
    """
    new_std = {}
    for kk, vv in out_bias.items():
        if vv.shape == out_std[kk].shape:
            new_std[kk] = out_std[kk]
        else:
            new_std[kk] = np.ones_like(vv)
    return out_bias, new_std


def _make_preset_out_bias(
    ntypes: int,
    ibias: list[np.ndarray | None],
) -> np.ndarray | None:
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


def _fill_stat_with_global(
    atomic_stat: np.ndarray | None,
    global_stat: np.ndarray,
) -> np.ndarray | None:
    """This function is used to fill atomic stat with global stat.

    Parameters
    ----------
    atomic_stat : Union[np.ndarray, None]
        The atomic stat.
    global_stat : np.ndarray
        The global stat.
    if the atomic stat is None, use global stat.
    if the atomic stat is not None, but has nan values (missing atypes), fill with global stat.
    """
    if atomic_stat is None:
        return global_stat
    else:
        atomic_stat = atomic_stat.reshape(*global_stat.shape)
        return np.nan_to_num(
            np.where(
                np.isnan(atomic_stat) & ~np.isnan(global_stat), global_stat, atomic_stat
            )
        )


def _compute_model_predict(
    sampled: list[dict],
    keys: list[str],
    model_forward: Callable,
) -> dict[str, list[np.ndarray]]:
    """Compute model predictions for all samples."""
    model_predict = {kk: [] for kk in keys}
    for system in sampled:
        # Convert inputs to numpy to avoid gradient issues
        coord = to_numpy_array(system["coord"])
        atype = to_numpy_array(system["atype"])
        box = to_numpy_array(system["box"])
        fparam = to_numpy_array(system.get("fparam", None))
        aparam = to_numpy_array(system.get("aparam", None))

        sample_predict = model_forward(coord, atype, box, fparam=fparam, aparam=aparam)
        for kk in keys:
            model_predict[kk].append(
                sample_predict[kk]  # already numpy from model_forward
            )
    return model_predict


def compute_output_stats(
    merged: Callable[[], list[dict]] | list[dict],
    ntypes: int,
    keys: str | list[str],
    stat_file_path: DPPath | None = None,
    rcond: float | None = None,
    preset_bias: dict[str, list[np.ndarray | None]] | None = None,
    model_forward: Callable | None = None,
    stats_distinguish_types: bool = True,
    intensive: bool = False,
) -> tuple[dict, dict]:
    """
    Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

    Parameters
    ----------
    merged : Union[Callable[[], list[dict]], list[dict]]
        - list[dict]: A list of data samples from various data systems.
            Each element, `merged[i]`, is a data dictionary containing `keys`: `np.ndarray`
            originating from the `i`-th data system.
        - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
            only when needed. Since the sampling process can be slow and memory-intensive,
            the lazy function helps by only sampling once.
    ntypes : int
        The number of atom types.
    keys : Union[str, list[str]]
        The keys of the output properties to compute statistics for.
    stat_file_path : DPPath, optional
        The path to the stat file.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    preset_bias : dict[str, list[Optional[np.ndarray]]], optional
        Specifying atomic energy contribution in vacuum. Given by key:value pairs.
        The value is a list specifying the bias. the elements can be None or np.ndarray of output shape.
        For example: [None, [2.]] means type 0 is not set, type 1 is set to [2.]
        The `set_davg_zero` key in the descriptor should be set.
    model_forward : Callable, optional
        The wrapped forward function of atomic model.
        If not None, the model will be utilized to generate the original energy prediction,
        which will be subtracted from the energy label of the data.
        The difference will then be used to calculate the delta complement energy bias for each type.
    stats_distinguish_types : bool, optional
        Whether to distinguish different element types in the statistics.
    intensive : bool, optional
        Whether the fitting target is intensive.
    """
    # normalize keys to list
    keys = [keys] if isinstance(keys, str) else keys
    assert isinstance(keys, list)

    # try to restore the bias from stat file
    bias_atom_e, std_atom_e = _restore_from_file(stat_file_path, keys)

    # failed to restore the bias from stat file. compute
    if bias_atom_e is None:
        # only get data once, sampled is a list of dict[str, np.ndarray]
        sampled = merged() if callable(merged) else merged

        # remove the keys that are not in the sample
        new_keys = [
            ii
            for ii in keys
            if (ii in sampled[0].keys()) or ("atom_" + ii in sampled[0].keys())
        ]
        keys = new_keys

        # compute model predictions if model_forward is provided
        if model_forward is not None:
            model_pred = _compute_model_predict(sampled, keys, model_forward)
        else:
            model_pred = None

        # split system based on label
        atomic_sampled_idx = defaultdict(list)
        global_sampled_idx = defaultdict(list)

        for kk in keys:
            for idx, system in enumerate(sampled):
                if (("find_atom_" + kk) in system) and (
                    system["find_atom_" + kk] > 0.0
                ):
                    atomic_sampled_idx[kk].append(idx)
                elif (("find_" + kk) in system) and (system["find_" + kk] > 0.0):
                    global_sampled_idx[kk].append(idx)
                else:
                    continue

        # use index to gather model predictions for the corresponding systems.
        model_pred_g = (
            {
                kk: [
                    np.sum(vv[idx], axis=1) for idx in global_sampled_idx[kk]
                ]  # sum atomic dim
                for kk, vv in model_pred.items()
            }
            if model_pred
            else None
        )
        model_pred_a = (
            {
                kk: [vv[idx] for idx in atomic_sampled_idx[kk]]
                for kk, vv in model_pred.items()
            }
            if model_pred
            else None
        )

        # concat all frames within those systems
        model_pred_g = (
            {
                kk: np.concatenate(model_pred_g[kk])
                for kk in model_pred_g.keys()
                if len(model_pred_g[kk]) > 0
            }
            if model_pred
            else None
        )
        model_pred_a = (
            {
                kk: np.concatenate(model_pred_a[kk])
                for kk in model_pred_a.keys()
                if len(model_pred_a[kk]) > 0
            }
            if model_pred
            else None
        )

        # compute stat
        bias_atom_g, std_atom_g = compute_output_stats_global(
            sampled,
            ntypes,
            keys,
            rcond,
            preset_bias,
            global_sampled_idx,
            stats_distinguish_types,
            intensive,
            model_pred_g,
        )
        bias_atom_a, std_atom_a = compute_output_stats_atomic(
            sampled,
            ntypes,
            keys,
            atomic_sampled_idx,
            model_pred_a,
        )

        # merge global/atomic bias
        bias_atom_e, std_atom_e = {}, {}
        for kk in keys:
            # use atomic bias whenever available
            if kk in bias_atom_a:
                bias_atom_e[kk] = bias_atom_a[kk]
                std_atom_e[kk] = std_atom_a[kk]
            else:
                bias_atom_e[kk] = None
                std_atom_e[kk] = None
            # use global bias to fill missing atomic bias
            if kk in bias_atom_g:
                bias_atom_e[kk] = _fill_stat_with_global(
                    bias_atom_e[kk], bias_atom_g[kk]
                )
                std_atom_e[kk] = _fill_stat_with_global(std_atom_e[kk], std_atom_g[kk])
            if (bias_atom_e[kk] is None) or (std_atom_e[kk] is None):
                raise RuntimeError("Fail to compute stat.")

        if stat_file_path is not None:
            _save_to_file(stat_file_path, bias_atom_e, std_atom_e)

    return bias_atom_e, std_atom_e


def compute_output_stats_global(
    sampled: list[dict],
    ntypes: int,
    keys: list[str],
    rcond: float | None = None,
    preset_bias: dict[str, list[np.ndarray | None]] | None = None,
    global_sampled_idx: dict | None = None,
    stats_distinguish_types: bool = True,
    intensive: bool = False,
    model_pred: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """This function only handle stat computation from reduced global labels."""
    # return directly if no global samples
    if global_sampled_idx is None or all(
        len(v) == 0 for v in global_sampled_idx.values()
    ):
        return {}, {}

    # get label dict from sample; for each key, only picking the system with global labels.
    outputs = {
        kk: [to_numpy_array(sampled[idx][kk]) for idx in global_sampled_idx.get(kk, [])]
        for kk in keys
    }

    natoms_key = "natoms"
    input_natoms = {
        kk: [
            to_numpy_array(sampled[idx][natoms_key])
            for idx in global_sampled_idx.get(kk, [])
        ]
        for kk in keys
    }

    # shape: (nframes, ndim)
    merged_output = {
        kk: np.concatenate(outputs[kk]) for kk in keys if len(outputs[kk]) > 0
    }
    # shape: (nframes, ntypes)
    merged_natoms = {
        kk: np.concatenate(input_natoms[kk])[:, 2:]
        for kk in keys
        if len(input_natoms[kk]) > 0
    }
    nf = {kk: merged_natoms[kk].shape[0] for kk in keys if kk in merged_natoms}

    if preset_bias is not None:
        assigned_atom_ener = {
            kk: _make_preset_out_bias(ntypes, preset_bias[kk])
            if kk in preset_bias.keys()
            else None
            for kk in keys
        }
    else:
        assigned_atom_ener = dict.fromkeys(keys)

    if model_pred is None:
        stats_input = merged_output
    else:
        # subtract the model bias and output the delta bias
        stats_input = {
            kk: merged_output[kk] - model_pred[kk].reshape(merged_output[kk].shape)
            for kk in keys
            if kk in merged_output
        }

    bias_atom_e = {}
    std_atom_e = {}
    for kk in keys:
        if kk in stats_input:
            if not stats_distinguish_types:
                bias_atom_e[kk], std_atom_e[kk] = (
                    compute_stats_do_not_distinguish_types(
                        stats_input[kk],
                        merged_natoms[kk],
                        assigned_bias=assigned_atom_ener[kk],
                        intensive=intensive,
                    )
                )
            else:
                bias_atom_e[kk], std_atom_e[kk] = compute_stats_from_redu(
                    stats_input[kk],
                    merged_natoms[kk],
                    assigned_bias=assigned_atom_ener[kk],
                    rcond=rcond,
                )
        else:
            # this key does not have global labels, skip it.
            continue
    bias_atom_e, std_atom_e = _post_process_stat(bias_atom_e, std_atom_e)

    # compute and log rmse
    def rmse(x: np.ndarray) -> float:
        return np.sqrt(np.mean(np.square(x)))

    if model_pred is None:
        unbias_e = {
            kk: merged_natoms[kk] @ bias_atom_e[kk].reshape(ntypes, -1)
            for kk in bias_atom_e.keys()
        }
    else:
        unbias_e = {
            kk: model_pred[kk].reshape(nf[kk], -1)
            + merged_natoms[kk] @ bias_atom_e[kk].reshape(ntypes, -1)
            for kk in bias_atom_e.keys()
        }
    atom_numbs = {kk: merged_natoms[kk].sum(-1) for kk in bias_atom_e.keys()}

    for kk in bias_atom_e.keys():
        rmse_ae = rmse(
            (unbias_e[kk].reshape(nf[kk], -1) - merged_output[kk].reshape(nf[kk], -1))
            / atom_numbs[kk][:, None]
        )
        log.info(
            f"RMSE of {kk} per atom after linear regression is: {rmse_ae} in the unit of {kk}."
        )
    return bias_atom_e, std_atom_e


def compute_output_stats_atomic(
    sampled: list[dict],
    ntypes: int,
    keys: list[str],
    atomic_sampled_idx: dict | None = None,
    model_pred: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute output statistics from atomic labels."""
    # return directly if no atomic samples
    if atomic_sampled_idx is None or all(
        len(v) == 0 for v in atomic_sampled_idx.values()
    ):
        return {}, {}

    # get label dict from sample; for each key, only picking the system with atomic labels.
    outputs = {
        kk: [
            to_numpy_array(sampled[idx]["atom_" + kk])
            for idx in atomic_sampled_idx.get(kk, [])
        ]
        for kk in keys
    }
    natoms = {
        kk: [
            to_numpy_array(sampled[idx]["atype"])
            for idx in atomic_sampled_idx.get(kk, [])
        ]
        for kk in keys
    }

    # reshape outputs [nframes, nloc * ndim] --> reshape to [nframes * nloc, 1, ndim] for concatenation
    # reshape natoms [nframes, nloc] --> reshape to [nframes * nolc, 1] for concatenation
    natoms = {k: [sys_v.reshape(-1, 1) for sys_v in v] for k, v in natoms.items()}
    outputs = {
        k: [
            sys.reshape(natoms[k][sys_idx].shape[0], 1, -1)
            for sys_idx, sys in enumerate(v)
        ]
        for k, v in outputs.items()
    }

    merged_output = {
        kk: np.concatenate(outputs[kk]) for kk in keys if len(outputs[kk]) > 0
    }
    merged_natoms = {
        kk: np.concatenate(natoms[kk]) for kk in keys if len(natoms[kk]) > 0
    }
    # reshape merged data to [nf, nloc, ndim]
    merged_output = {
        kk: merged_output[kk].reshape((*merged_natoms[kk].shape, -1))
        for kk in merged_output
    }

    if model_pred is None:
        stats_input = merged_output
    else:
        # subtract the model bias and output the delta bias
        stats_input = {
            kk: merged_output[kk] - model_pred[kk].reshape(*merged_output[kk].shape)
            for kk in keys
            if kk in merged_output
        }

    bias_atom_e = {}
    std_atom_e = {}

    for kk in keys:
        if kk in stats_input:
            bias_atom_e[kk], std_atom_e[kk] = compute_stats_from_atomic(
                stats_input[kk],
                merged_natoms[kk],
            )
            # correction for missing types
            missing_types = ntypes - merged_natoms[kk].max() - 1
            if missing_types > 0:
                assert bias_atom_e[kk].dtype is std_atom_e[kk].dtype, (
                    "bias and std should be of the same dtypes"
                )
                nan_padding = np.empty(
                    (missing_types, bias_atom_e[kk].shape[1]),
                    dtype=bias_atom_e[kk].dtype,
                )
                nan_padding.fill(np.nan)
                bias_atom_e[kk] = np.concatenate([bias_atom_e[kk], nan_padding], axis=0)
                std_atom_e[kk] = np.concatenate([std_atom_e[kk], nan_padding], axis=0)
        else:
            # this key does not have atomic labels, skip it.
            continue
    return bias_atom_e, std_atom_e
