# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.common import (
    get_hash,
)
from deepmd.utils.path import (
    DPPath,
)


def _descriptor_rcut_smth(descrpt: Any) -> float:
    if hasattr(descrpt, "rcut_smth"):
        return descrpt.rcut_smth
    return descrpt.rcut_r_smth


def _descriptor_sel(descrpt: Any, last_dim: int) -> list[int]:
    if hasattr(descrpt, "get_sel"):
        sel = descrpt.get_sel()
    elif last_dim == 1:
        sel = descrpt.sel_r
    else:
        sel = descrpt.sel_a
    if isinstance(sel, np.ndarray):
        sel = sel.tolist()
    elif isinstance(sel, int):
        sel = [sel]
    return [int(ii) for ii in sel]


def _descriptor_stat_path(
    descrpt: Any,
    stat_file_path: DPPath | None,
    last_dim: int,
    mixed_types: bool,
) -> DPPath | None:
    if stat_file_path is None:
        return None
    sel = _descriptor_sel(descrpt, last_dim)
    stat_hash = get_hash(
        {
            "type": "se_a" if last_dim == 4 else "se_r",
            "ntypes": descrpt.get_ntypes(),
            "rcut": round(descrpt.get_rcut(), 2),
            "rcut_smth": round(_descriptor_rcut_smth(descrpt), 2),
            "nsel": sum(sel),
            "sel": sel,
            "mixed_types": mixed_types,
        }
    )
    return stat_file_path / stat_hash


def _stat_keys(ntypes: int, angular: bool) -> list[str]:
    keys = [f"r_{ii}" for ii in range(ntypes)]
    if angular:
        keys.extend(f"a_{ii}" for ii in range(ntypes))
    return keys


def _load_se_input_stats(
    path: DPPath | None,
    ntypes: int,
    angular: bool,
) -> dict[str, list[list[float]]] | None:
    if path is None or not path.is_dir():
        return None
    if any(not (path / kk).is_file() for kk in _stat_keys(ntypes, angular)):
        return None

    sumr = []
    sumn = []
    sumr2 = []
    suma = []
    suma2 = []
    for type_i in range(ntypes):
        r_stat = (path / f"r_{type_i}").load_numpy()
        sumn.append(float(r_stat[0]))
        sumr.append(float(r_stat[1]))
        sumr2.append(float(r_stat[2]))
        if angular:
            a_stat = (path / f"a_{type_i}").load_numpy()
            suma.append(float(a_stat[1]) / 3.0)
            suma2.append(float(a_stat[2]) / 3.0)

    ret = {
        "sumr": [sumr],
        "sumn": [sumn],
        "sumr2": [sumr2],
    }
    if angular:
        ret["suma"] = [suma]
        ret["suma2"] = [suma2]
    return ret


def _save_se_input_stats(
    path: DPPath | None,
    stat_dict: dict[str, Any],
    ntypes: int,
    angular: bool,
) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)

    sumr = np.sum(stat_dict["sumr"], axis=0)
    sumn = np.sum(stat_dict["sumn"], axis=0)
    sumr2 = np.sum(stat_dict["sumr2"], axis=0)
    if angular:
        suma = np.sum(stat_dict["suma"], axis=0)
        suma2 = np.sum(stat_dict["suma2"], axis=0)

    for type_i in range(ntypes):
        (path / f"r_{type_i}").save_numpy(
            np.array([sumn[type_i], sumr[type_i], sumr2[type_i]])
        )
        if angular:
            (path / f"a_{type_i}").save_numpy(
                np.array([3.0 * sumn[type_i], 3.0 * suma[type_i], 3.0 * suma2[type_i]])
            )


def load_or_compute_se_input_stats(
    descrpt: Any,
    stat_file_path: DPPath | None,
    last_dim: int,
    compute: Callable[[], dict[str, Any]],
    mixed_types: bool = False,
) -> dict[str, Any]:
    """Load or compute SE descriptor input statistics using EnvMatStatSe format."""
    angular = last_dim == 4
    stat_path = _descriptor_stat_path(descrpt, stat_file_path, last_dim, mixed_types)
    stat_dict = _load_se_input_stats(stat_path, descrpt.get_ntypes(), angular)
    if stat_dict is not None:
        return stat_dict

    stat_dict = compute()
    _save_se_input_stats(stat_path, stat_dict, descrpt.get_ntypes(), angular)
    return stat_dict
