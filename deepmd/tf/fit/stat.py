# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np

from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)


def load_param_stats(
    stat_file_path: DPPath | None,
    name: str,
    dim: int,
) -> list[StatItem] | None:
    if (
        stat_file_path is None
        or not stat_file_path.is_dir()
        or not (stat_file_path / name).is_file()
    ):
        return None
    arr = (stat_file_path / name).load_numpy()
    if arr.shape != (dim, 3):
        raise ValueError(f"Invalid {name} stat shape {arr.shape}; expected ({dim}, 3).")
    return [
        StatItem(number=arr[ii, 0], sum=arr[ii, 1], squared_sum=arr[ii, 2])
        for ii in range(dim)
    ]


def save_param_stats(
    stat_file_path: DPPath | None,
    name: str,
    stats: list[StatItem],
) -> None:
    if stat_file_path is None:
        return
    stat_file_path.mkdir(parents=True, exist_ok=True)
    arr = np.array([[ss.number, ss.sum, ss.squared_sum] for ss in stats])
    (stat_file_path / name).save_numpy(arr)


def make_fparam_stats(all_stat: dict[str, Any], dim: int) -> list[StatItem]:
    cat_data = np.concatenate(all_stat["fparam"], axis=0)
    cat_data = np.reshape(cat_data, [-1, dim])
    sumv = np.sum(cat_data, axis=0)
    sumv2 = np.sum(cat_data * cat_data, axis=0)
    sumn = cat_data.shape[0]
    return [
        StatItem(number=sumn, sum=sumv[ii], squared_sum=sumv2[ii]) for ii in range(dim)
    ]


def make_aparam_stats(all_stat: dict[str, Any], dim: int) -> list[StatItem]:
    sys_sumv = []
    sys_sumv2 = []
    sys_sumn = []
    for ss_ in all_stat["aparam"]:
        ss = np.reshape(ss_, [-1, dim])
        sys_sumv.append(np.sum(ss, axis=0))
        sys_sumv2.append(np.sum(ss * ss, axis=0))
        sys_sumn.append(ss.shape[0])
    sumv = np.sum(sys_sumv, axis=0)
    sumv2 = np.sum(sys_sumv2, axis=0)
    sumn = np.sum(sys_sumn)
    return [
        StatItem(number=sumn, sum=sumv[ii], squared_sum=sumv2[ii]) for ii in range(dim)
    ]


def stats_avg_std(
    stats: list[StatItem],
    protection: float,
) -> tuple[np.ndarray, np.ndarray]:
    avg = np.array([ss.compute_avg() for ss in stats])
    std = np.array([ss.compute_std(protection=protection) for ss in stats])
    return avg, std
