#!/usr/bin/env python3
"""Quickly create a configuration file for smooth model."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

__all__ = ["config"]


DEFAULT_DATA: Dict[str, Any] = {
    "use_smooth": True,
    "sel_a": [],
    "rcut_smth": -1,
    "rcut": -1,
    "filter_neuron": [20, 40, 80],
    "filter_resnet_dt": False,
    "axis_neuron": 8,
    "fitting_neuron": [240, 240, 240],
    "fitting_resnet_dt": True,
    "coord_norm": True,
    "type_fitting_net": False,
    "systems": [],
    "set_prefix": "set",
    "stop_batch": -1,
    "batch_size": -1,
    "start_lr": 0.001,
    "decay_steps": -1,
    "decay_rate": 0.95,
    "start_pref_e": 0.02,
    "limit_pref_e": 1,
    "start_pref_f": 1000,
    "limit_pref_f": 1,
    "start_pref_v": 0,
    "limit_pref_v": 0,
    "seed": 1,
    "disp_file": "lcurve.out",
    "disp_freq": 1000,
    "numb_test": 10,
    "save_freq": 10000,
    "save_ckpt": "model.ckpt",
    "disp_training": True,
    "time_training": True,
}


def valid_dir(path: Path):
    """Check if directory is a valid deepmd system directory.

    Parameters
    ----------
    path : Path
        path to directory

    Raises
    ------
    OSError
        if `type.raw` is missing on dir or `box.npy` or `coord.npy` are missing in one
        of the sets subdirs
    """
    if not (path / "type.raw").is_file():
        raise OSError
    for ii in path.glob("set.*"):
        if not (ii / "box.npy").is_file():
            raise OSError
        if not (ii / "coord.npy").is_file():
            raise OSError


def load_systems(dirs: List[Path]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load systems to memory for disk.

    Parameters
    ----------
    dirs : List[Path]
        list of system directories paths

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        atoms types and structure cells formated as Nx9 array
    """
    all_type = []
    all_box = []
    for d in dirs:
        sys_type = np.loadtxt(d / "type.raw", dtype=int)
        sys_box = np.vstack([np.load(s / "box.npy") for s in d.glob("set.*")])
        all_type.append(sys_type)
        all_box.append(sys_box)
    return all_type, all_box


def get_system_names() -> List[Path]:
    """Get system directory paths from stdin.

    Returns
    -------
    List[Path]
        list of system directories paths
    """
    dirs = input("Enter system path(s) (seperated by space, wild card supported): \n")
    system_dirs = []
    for dir_str in dirs.split():
        found_dirs = Path.cwd().glob(dir_str)
        for d in found_dirs:
            valid_dir(d)
            system_dirs.append(d)

    return system_dirs


def get_rcut() -> float:
    """Get rcut from stdin from user.

    Returns
    -------
    float
        input rcut lenght converted to float

    Raises
    ------
    ValueError
        if rcut is smaller than 0.0
    """
    dv = 6.0
    rcut_input = input(f"Enter rcut (default {dv:.1f} A): \n")
    try:
        rcut = float(rcut_input)
    except ValueError as e:
        print(f"invalid rcut: {e} setting to default: {dv:.1f}")
        rcut = dv
    if rcut <= 0:
        raise ValueError("rcut should be > 0")
    return rcut


def get_batch_size_rule() -> int:
    """Get minimal batch size from user from stdin.

    Returns
    -------
    int
        size of the batch

    Raises
    ------
    ValueError
        if batch size is <= 0
    """
    dv = 32
    matom_input = input(
        f"Enter the minimal number of atoms in a batch (default {dv:d}: \n"
    )
    try:
        matom = int(matom_input)
    except ValueError as e:
        print(f"invalid batch size: {e} setting to default: {dv:d}")
        matom = dv
    if matom <= 0:
        raise ValueError("the number should be > 0")
    return matom


def get_stop_batch() -> int:
    """Get stop batch from user from stdin.

    Returns
    -------
    int
        size of the batch

    Raises
    ------
    ValueError
        if stop batch is <= 0
    """
    dv = 1000000
    sb_input = input(f"Enter the stop batch (default {dv:d}): \n")
    try:
        sb = int(sb_input)
    except ValueError as e:
        print(f"invalid stop batch: {e} setting to default: {dv:d}")
        sb = dv
    if sb <= 0:
        raise ValueError("the number should be > 0")
    return sb


def get_ntypes(all_type: List[np.ndarray]) -> int:
    """Count number of unique elements.

    Parameters
    ----------
    all_type : List[np.ndarray]
        list with arrays specifying elements of structures

    Returns
    -------
    int
        number of unique elements
    """
    return len(np.unique(all_type))


def get_max_density(
    all_type: List[np.ndarray], all_box: List[np.ndarray]
) -> np.ndarray:
    """Compute maximum density in suppliedd cells.

    Parameters
    ----------
    all_type : List[np.ndarray]
        list with arrays specifying elements of structures
    all_box : List[np.ndarray]
        list with arrays specifying cells for all structures

    Returns
    -------
    float
        maximum atom density in all supplies structures for each element individually
    """
    ntypes = get_ntypes(all_type)
    all_max = []
    for tt, bb in zip(all_type, all_box):
        vv = np.reshape(bb, [-1, 3, 3])
        vv = np.linalg.det(vv)
        min_v = np.min(vv)
        type_count = []
        for ii in range(ntypes):
            type_count.append(sum(tt == ii))
        max_den = type_count / min_v
        all_max.append(max_den)
    all_max = np.max(all_max, axis=0)
    return all_max


def suggest_sel(
    all_type: List[np.ndarray],
    all_box: List[np.ndarray],
    rcut: float,
    ratio: float = 1.5,
) -> List[int]:
    """Suggest selection parameter.

    Parameters
    ----------
    all_type : List[np.ndarray]
        list with arrays specifying elements of structures
    all_box : List[np.ndarray]
        list with arrays specifying cells for all structures
    rcut : float
        cutoff radius
    ratio : float, optional
        safety margin to add to estimated value, by default 1.5

    Returns
    -------
    List[int]
        [description]
    """
    max_den = get_max_density(all_type, all_box)
    return [int(ii) for ii in max_den * 4.0 / 3.0 * np.pi * rcut ** 3 * ratio]


def suggest_batch_size(all_type: List[np.ndarray], min_atom: int) -> List[int]:
    """Get suggestion for batch size.

    Parameters
    ----------
    all_type : List[np.ndarray]
        list with arrays specifying elements of structures
    min_atom : int
        minimal number of atoms in batch

    Returns
    -------
    List[int]
        suggested batch sizes for each system
    """
    bs = []
    for ii in all_type:
        natoms = len(ii)
        tbs = min_atom // natoms
        if (min_atom // natoms) * natoms != min_atom:
            tbs += 1
        bs.append(tbs)
    return bs


def suggest_decay(stop_batch: int) -> Tuple[int, float]:
    """Suggest number of decay steps and decay rate.

    Parameters
    ----------
    stop_batch : int
        stop batch number

    Returns
    -------
    Tuple[int, float]
        number of decay steps and decay rate
    """
    decay_steps = int(stop_batch // 200)
    decay_rate = 0.95
    return decay_steps, decay_rate


def config(*, output: str, **kwargs):
    """Auto config file generator.

    Parameters
    ----------
    output: str
        file to write config file

    Raises
    ------
    RuntimeError
        if user does not input any systems
    ValueError
        if output file is of wrong type
    """
    all_sys = get_system_names()
    if len(all_sys) == 0:
        raise RuntimeError("no system specified")
    rcut = get_rcut()
    matom = get_batch_size_rule()
    stop_batch = get_stop_batch()

    all_type, all_box = load_systems(all_sys)
    sel = suggest_sel(all_type, all_box, rcut, ratio=1.5)
    bs = suggest_batch_size(all_type, matom)
    decay_steps, decay_rate = suggest_decay(stop_batch)

    jdata = DEFAULT_DATA.copy()
    jdata["systems"] = [str(ii) for ii in all_sys]
    jdata["sel_a"] = sel
    jdata["rcut"] = rcut
    jdata["rcut_smth"] = rcut - 0.2
    jdata["stop_batch"] = stop_batch
    jdata["batch_size"] = bs
    jdata["decay_steps"] = decay_steps
    jdata["decay_rate"] = decay_rate

    with open(output, "w") as fp:
        if output.endswith("json"):
            json.dump(jdata, fp, indent=4)
        elif output.endswith(("yml", "yaml")):
            yaml.safe_dump(jdata, fp, default_flow_style=False)
        else:
            raise ValueError("output file must be of type json or yaml")
