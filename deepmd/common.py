# SPDX-License-Identifier: LGPL-3.0-or-later
import glob
import json
import os
import platform
import shutil
import warnings
from hashlib import (
    sha1,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    get_args,
)

try:
    from typing import Literal  # python >=3.8
except ImportError:
    from typing_extensions import Literal  # type: ignore

import numpy as np
import yaml

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.path import (
    DPPath,
)

__all__ = [
    "VALID_ACTIVATION",
    "VALID_PRECISION",
    "expand_sys_str",
    "get_np_precision",
    "j_loader",
    "make_default_mesh",
    "select_idx_map",
]

_PRECISION = Literal["default", "float16", "float32", "float64"]
_ACTIVATION = Literal[
    "relu",
    "relu6",
    "softplus",
    "sigmoid",
    "tanh",
    "gelu",
    "gelu_tf",
    "silu",
    "silut",
    "none",
    "linear",
]
# get_args is new in py38
VALID_PRECISION: set[_PRECISION] = set(get_args(_PRECISION))
VALID_ACTIVATION: set[_ACTIVATION] = set(get_args(_ACTIVATION))

if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    __all__ += [
        "_ACTIVATION",
        "_DICT_VAL",
        "_PRECISION",
    ]


def select_idx_map(atom_types: np.ndarray, select_types: np.ndarray) -> np.ndarray:
    """Build map of indices for element supplied element types from all atoms list.

    Parameters
    ----------
    atom_types : np.ndarray
        array specifying type for each atoms as integer
    select_types : np.ndarray
        types of atoms you want to find indices for

    Returns
    -------
    np.ndarray
        indices of types of atoms defined by `select_types` in `atom_types` array

    Warnings
    --------
    `select_types` array will be sorted before finding indices in `atom_types`
    """
    sort_select_types = np.sort(select_types)
    idx_map = []
    for ii in sort_select_types:
        idx_map.append(np.where(atom_types == ii)[0])
    return np.concatenate(idx_map)


def make_default_mesh(pbc: bool, mixed_type: bool) -> np.ndarray:
    """Make mesh.

    Only the size of mesh matters, not the values:
    * 6 for PBC, no mixed types
    * 0 for no PBC, no mixed types
    * 7 for PBC, mixed types
    * 1 for no PBC, mixed types

    Parameters
    ----------
    pbc : bool
        if True, the mesh will be made for periodic boundary conditions
    mixed_type : bool
        if True, the mesh will be made for mixed types

    Returns
    -------
    np.ndarray
        mesh
    """
    mesh_size = int(pbc) * 6 + int(mixed_type)
    default_mesh = np.zeros(mesh_size, dtype=np.int32)
    return default_mesh


def j_deprecated(
    jdata: dict[str, "_DICT_VAL"], key: str, deprecated_key: list[str] = []
) -> "_DICT_VAL":
    """Assert that supplied dictionary contains specified key.

    Parameters
    ----------
    jdata : dict[str, _DICT_VAL]
        dictionary to check
    key : str
        key to check
    deprecated_key : list[str], optional
        list of deprecated keys, by default []

    Returns
    -------
    _DICT_VAL
        value that was store unde supplied key

    Raises
    ------
    RuntimeError
        if the key is not present
    """
    if key not in jdata.keys():
        for ii in deprecated_key:
            if ii in jdata.keys():
                warnings.warn(f"the key {ii} is deprecated, please use {key} instead")
                return jdata[ii]
        else:
            raise RuntimeError(f"json database must provide key {key}")
    else:
        return jdata[key]


def j_loader(filename: Union[str, Path]) -> dict[str, Any]:
    """Load yaml or json settings file.

    Parameters
    ----------
    filename : Union[str, Path]
        path to file

    Returns
    -------
    dict[str, Any]
        loaded dictionary

    Raises
    ------
    TypeError
        if the supplied file is of unsupported type
    """
    filepath = Path(filename)
    if filepath.suffix.endswith("json"):
        with filepath.open() as fp:
            return json.load(fp)
    elif filepath.suffix.endswith(("yml", "yaml")):
        with filepath.open() as fp:
            return yaml.safe_load(fp)
    else:
        raise TypeError("config file must be json, or yaml/yml")


def expand_sys_str(root_dir: Union[str, Path]) -> list[str]:
    """Recursively iterate over directories taking those that contain `type.raw` file.

    Parameters
    ----------
    root_dir : Union[str, Path]
        starting directory

    Returns
    -------
    list[str]
        list of string pointing to system directories
    """
    root_dir = DPPath(root_dir)
    matches = [str(d) for d in root_dir.rglob("*") if (d / "type.raw").is_file()]
    if (root_dir / "type.raw").is_file():
        matches.append(str(root_dir))
    return matches


def rglob_sys_str(root_dir: str, patterns: list[str]) -> list[str]:
    """Recursively iterate over directories taking those that contain `type.raw` file.

    Parameters
    ----------
    root_dir : str, Path
        starting directory
    patterns : list[str]
        list of glob patterns to match directories

    Returns
    -------
    list[str]
        list of string pointing to system directories
    """
    root_dir = Path(root_dir)
    matches = []
    for pattern in patterns:
        matches.extend(
            [str(d) for d in root_dir.rglob(pattern) if (d / "type.raw").is_file()]
        )
    return list(set(matches))  # remove duplicates


def get_np_precision(precision: "_PRECISION") -> np.dtype:
    """Get numpy precision constant from string.

    Parameters
    ----------
    precision : _PRECISION
        string name of numpy constant or default

    Returns
    -------
    np.dtype
        numpy precision constant

    Raises
    ------
    RuntimeError
        if string is invalid
    """
    if precision == "default":
        return GLOBAL_NP_FLOAT_PRECISION
    elif precision == "float16":
        return np.float16
    elif precision == "float32":
        return np.float32
    elif precision == "float64":
        return np.float64
    else:
        raise RuntimeError(f"{precision} is not a valid precision")


def symlink_prefix_files(old_prefix: str, new_prefix: str) -> None:
    """Create symlinks from old checkpoint prefix to new one.

    On Windows this function will copy files instead of creating symlinks.

    Parameters
    ----------
    old_prefix : str
        old checkpoint prefix, all files with this prefix will be symlinked
    new_prefix : str
        new checkpoint prefix
    """
    original_files = glob.glob(old_prefix + ".*")
    for ori_ff in original_files:
        new_ff = new_prefix + ori_ff[len(old_prefix) :]
        try:
            # remove old one
            os.remove(new_ff)
        except OSError:
            pass
        if platform.system() != "Windows":
            # by default one does not have access to create symlink on Windows
            os.symlink(os.path.relpath(ori_ff, os.path.dirname(new_ff)), new_ff)
        else:
            shutil.copyfile(ori_ff, new_ff)


def get_hash(obj) -> str:
    """Get hash of object.

    Parameters
    ----------
    obj
        object to hash
    """
    return sha1(json.dumps(obj).encode("utf-8")).hexdigest()


def j_get_type(data: dict, class_name: str = "object") -> str:
    """Get the type from the data.

    Parameters
    ----------
    data : dict
        the data
    class_name : str, optional
        the name of the class for error message, by default "object"

    Returns
    -------
    str
        the type
    """
    try:
        return data["type"]
    except KeyError as e:
        raise KeyError(f"the type of the {class_name} should be set by `type`") from e
