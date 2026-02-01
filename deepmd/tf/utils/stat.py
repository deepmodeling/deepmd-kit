# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Optional,
)

import numpy as np

from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)
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


def _post_process_stat(
    out_bias,
    out_std,
):
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


def compute_output_stats(
    all_stat: dict,
    ntypes: int,
    keys: list[str] = ["energy"],
    stat_file_path: Optional[DPPath] = None,
    rcond: Optional[float] = None,
    mixed_type: bool = False,
) -> tuple[dict, dict]:
    """Compute output statistics for TensorFlow models.

    This function is designed to be compatible with the PyTorch backend
    to ensure consistent stat file formats and values.

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
        Computed bias values with shape (ntypes, 1) for compatibility
    std_out : dict
        Computed standard deviation values with shape (ntypes, 1) for compatibility
    """
    # Try to restore from file first
    bias_out, std_out = _restore_from_file(stat_file_path, keys)

    if bias_out is not None and std_out is not None:
        log.info("Successfully restored statistics from stat file")
        return bias_out, std_out

    # If restore failed, compute from data
    log.info("Computing statistics from training data")

    bias_out = {}
    std_out = {}

    for key in keys:
        if key in all_stat:
            # Get energy and natoms data
            energy_data = np.concatenate(all_stat[key])
            natoms_vec = np.concatenate(all_stat["natoms_vec"])

            # Calculate the number of frames and elements per frame
            nframes = energy_data.shape[0]
            elements_per_frame = natoms_vec.shape[0] // nframes

            # Reshape natoms_vec to (nframes, elements_per_frame) then take type columns
            if natoms_vec.ndim == 1:
                # Reshape the 1D concatenated data into frames
                natoms_data = natoms_vec.reshape(nframes, elements_per_frame)[:, 2:]
            else:
                # Already 2D, slice directly
                natoms_data = natoms_vec[:, 2:]

            # Ensure we have the right number of types
            if natoms_data.shape[1] != ntypes:
                raise ValueError(
                    f"Mismatch between ntypes ({ntypes}) and natoms data shape ({natoms_data.shape[1]})"
                )

            # Compute statistics using existing utility
            bias, std = compute_stats_from_redu(
                energy_data.reshape(-1, 1),  # Reshape to column vector
                natoms_data,
                rcond=rcond,
            )

            # Reshape outputs to match PyTorch format: (ntypes, 1)
            bias_out[key] = bias.reshape(ntypes, 1)

            # For std, we initially get a scalar from compute_stats_from_redu.
            # To match PyTorch behavior exactly, we use the post-processing logic
            # that sets std to ones when shape doesn't match bias shape.
            std_out[key] = std.reshape(1, 1)  # First reshape to (1, 1)

            log.info(
                f"Statistics computed for {key}: bias shape {bias_out[key].shape}, std shape {std_out[key].shape}"
            )

    # Apply post-processing to match PyTorch behavior exactly
    bias_out, std_out = _post_process_stat(bias_out, std_out)

    # Save to file if path provided
    if stat_file_path is not None and bias_out:
        _save_to_file(stat_file_path, bias_out, std_out)
        log.info("Statistics saved to stat file")

    return bias_out, std_out
