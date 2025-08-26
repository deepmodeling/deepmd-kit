# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Optional,
)

import numpy as np

from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)


def _restore_from_file(
    stat_file_path: DPPath,
    keys: list[str] = ["energy"],
) -> Optional[tuple[dict, dict]]:
    """Restore bias and std from stat file.

    Parameters
    ----------
    stat_file_path : DPPath
        Path to the stat file directory/file
    keys : list[str]
        Keys to restore statistics for

    Returns
    -------
    ret_bias : dict or None
        Bias values for each key
    ret_std : dict or None
        Standard deviation values for each key
    """
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
) -> None:
    """Save bias and std to stat file.

    Parameters
    ----------
    stat_file_path : DPPath
        Path to the stat file directory/file
    bias_out : dict
        Bias values for each key
    std_out : dict
        Standard deviation values for each key
    """
    assert stat_file_path is not None
    stat_file_path.mkdir(exist_ok=True, parents=True)
    for kk, vv in bias_out.items():
        fp = stat_file_path / f"bias_atom_{kk}"
        fp.save_numpy(vv)
    for kk, vv in std_out.items():
        fp = stat_file_path / f"std_atom_{kk}"
        fp.save_numpy(vv)


def compute_output_stats(
    all_stat: dict,
    ntypes: int,
    keys: list[str] = ["energy"],
    stat_file_path: Optional[DPPath] = None,
    rcond: Optional[float] = None,
    mixed_type: bool = False,
) -> tuple[dict, dict]:
    """Compute output statistics for TensorFlow models.

    This is a simplified version of the PyTorch compute_output_stats function
    adapted for TensorFlow models.

    Parameters
    ----------
    all_stat : dict
        Dictionary containing statistical data
    ntypes : int
        Number of atom types
    keys : list[str]
        Keys to compute statistics for
    stat_file_path : DPPath, optional
        Path to save/load statistics
    rcond : float, optional
        Condition number for regression
    mixed_type : bool
        Whether mixed type format is used

    Returns
    -------
    bias_out : dict
        Computed bias values
    std_out : dict
        Computed standard deviation values
    """
    # Try to restore from file first
    bias_out, std_out = _restore_from_file(stat_file_path, keys)

    if bias_out is not None and std_out is not None:
        log.info("Successfully restored statistics from stat file")
        return bias_out, std_out

    # If restore failed, compute from data
    log.info("Computing statistics from training data")

    from deepmd.utils.out_stat import (
        compute_stats_from_redu,
    )

    bias_out = {}
    std_out = {}

    for key in keys:
        if key in all_stat:
            # Get energy and natoms data
            energy_data = np.concatenate(all_stat[key])
            natoms_data = np.concatenate(all_stat["natoms_vec"])[
                :, 2:
            ]  # Skip first 2 elements

            # Compute statistics using existing utility
            bias, std = compute_stats_from_redu(
                energy_data.reshape(-1, 1),  # Reshape to column vector
                natoms_data,
                rcond=rcond,
            )

            bias_out[key] = bias.reshape(-1)  # Flatten to 1D
            std_out[key] = std.reshape(-1)  # Flatten to 1D

            log.info(
                f"Statistics computed for {key}: bias shape {bias_out[key].shape}, std shape {std_out[key].shape}"
            )

    # Save to file if path provided
    if stat_file_path is not None and bias_out:
        _save_to_file(stat_file_path, bias_out, std_out)
        log.info("Statistics saved to stat file")

    return bias_out, std_out
