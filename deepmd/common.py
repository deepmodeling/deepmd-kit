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
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
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
    "data_requirement",
    "add_data_requirement",
    "select_idx_map",
    "make_default_mesh",
    "j_must_have",
    "j_loader",
    "expand_sys_str",
    "get_np_precision",
]


if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    _PRECISION = Literal["default", "float16", "float32", "float64"]
    _ACTIVATION = Literal[
        "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu", "gelu_tf"
    ]
    __all__.extend(
        [
            "_DICT_VAL",
            "_PRECISION",
            "_ACTIVATION",
        ]
    )


# TODO this is not a good way to do things. This is some global variable to which
# TODO anyone can write and there is no good way to keep track of the changes
data_requirement = {}


def add_data_requirement(
    key: str,
    ndof: int,
    atomic: bool = False,
    must: bool = False,
    high_prec: bool = False,
    type_sel: Optional[bool] = None,
    repeat: int = 1,
    default: float = 0.0,
    dtype: Optional[np.dtype] = None,
    output_natoms_for_type_sel: bool = False,
):
    """Specify data requirements for training.

    Parameters
    ----------
    key : str
        type of data stored in corresponding `*.npy` file e.g. `forces` or `energy`
    ndof : int
        number of the degrees of freedom, this is tied to `atomic` parameter e.g. forces
        have `atomic=True` and `ndof=3`
    atomic : bool, optional
        specifies whwther the `ndof` keyworrd applies to per atom quantity or not,
        by default False
    must : bool, optional
        specifi if the `*.npy` data file must exist, by default False
    high_prec : bool, optional
        if true load data to `np.float64` else `np.float32`, by default False
    type_sel : bool, optional
        select only certain type of atoms, by default None
    repeat : int, optional
        if specify repaeat data `repeat` times, by default 1
    default : float, optional, default=0.
        default value of data
    dtype : np.dtype, optional
        the dtype of data, overwrites `high_prec` if provided
    output_natoms_for_type_sel : bool, optional
        if True and type_sel is True, the atomic dimension will be natoms instead of nsel
    """
    data_requirement[key] = {
        "ndof": ndof,
        "atomic": atomic,
        "must": must,
        "high_prec": high_prec,
        "type_sel": type_sel,
        "repeat": repeat,
        "default": default,
        "dtype": dtype,
        "output_natoms_for_type_sel": output_natoms_for_type_sel,
    }


def select_idx_map(atom_types: np.ndarray, select_types: np.ndarray) -> np.ndarray:
    """Build map of indices for element supplied element types from all atoms list.

    Parameters
    ----------
    atom_types : np.ndarray
        array specifing type for each atoms as integer
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


# TODO maybe rename this to j_deprecated and only warn about deprecated keys,
# TODO if the deprecated_key argument is left empty function puppose is only custom
# TODO error since dict[key] already raises KeyError when the key is missing
def j_must_have(
    jdata: Dict[str, "_DICT_VAL"], key: str, deprecated_key: List[str] = []
) -> "_DICT_VAL":
    """Assert that supplied dictionary conaines specified key.

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


def j_loader(filename: Union[str, Path]) -> Dict[str, Any]:
    """Load yaml or json settings file.

    Parameters
    ----------
    filename : Union[str, Path]
        path to file

    Returns
    -------
    Dict[str, Any]
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


# TODO port completely to pathlib when all callers are ported
def expand_sys_str(root_dir: Union[str, Path]) -> List[str]:
    """Recursively iterate over directories taking those that contain `type.raw` file.

    Parameters
    ----------
    root_dir : Union[str, Path]
        starting directory

    Returns
    -------
    List[str]
        list of string pointing to system directories
    """
    root_dir = DPPath(root_dir)
    matches = [str(d) for d in root_dir.rglob("*") if (d / "type.raw").is_file()]
    if (root_dir / "type.raw").is_file():
        matches.append(str(root_dir))
    return matches


def get_np_precision(precision: "_PRECISION") -> np.dtype:
    """Get numpy precision constant from string.

    Parameters
    ----------
    precision : _PRECISION
        string name of numpy constant or default

    Returns
    -------
    np.dtype
        numpy presicion constant

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


def symlink_prefix_files(old_prefix: str, new_prefix: str):
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
