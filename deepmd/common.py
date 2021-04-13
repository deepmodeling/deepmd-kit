"""Collection of functions and classes used throughout the whole package."""

import json
import warnings
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import yaml

from deepmd.env import op_module, tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, GLOBAL_NP_FLOAT_PRECISION

if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    _OBJ = TypeVar("_OBJ")
    try:
        from typing import Literal  # python >3.6
    except ImportError:
        from typing_extensions import Literal  # type: ignore
    _ACTIVATION = Literal["relu", "relu6", "softplus", "sigmoid", "tanh", "gelu"]
    _PRECISION = Literal["default", "float16", "float32", "float64"]

# define constants
PRECISION_DICT = {
    "default": GLOBAL_TF_FLOAT_PRECISION,
    "float16": tf.float16,
    "float32": tf.float32,
    "float64": tf.float64,
}


def gelu(x: tf.Tensor) -> tf.Tensor:
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.

    Parameters
    ----------
    x : tf.Tensor
        float Tensor to perform activation

    Returns
    -------
    `x` with the GELU activation applied

    References
    ----------
    Original paper
    https://arxiv.org/abs/1606.08415
    """
    return op_module.gelu(x)


# TODO this is not a good way to do things. This is some global variable to which
# TODO anyone can write and there is no good way to keep track of the changes
data_requirement = {}

ACTIVATION_FN_DICT = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "softplus": tf.nn.softplus,
    "sigmoid": tf.sigmoid,
    "tanh": tf.nn.tanh,
    "gelu": gelu,
}


def add_data_requirement(
    key: str,
    ndof: int,
    atomic: bool = False,
    must: bool = False,
    high_prec: bool = False,
    type_sel: bool = None,
    repeat: int = 1,
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
        if tru load data to `np.float64` else `np.float32`, by default False
    type_sel : bool, optional
        select only certain type of atoms, by default None
    repeat : int, optional
        if specify repaeat data `repeat` times, by default 1
    """
    data_requirement[key] = {
        "ndof": ndof,
        "atomic": atomic,
        "must": must,
        "high_prec": high_prec,
        "type_sel": type_sel,
        "repeat": repeat,
    }


def select_idx_map(
    atom_types: np.ndarray, select_types: np.ndarray
) -> np.ndarray:
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
    idx_map = np.array([], dtype=int)
    for ii in sort_select_types:
        idx_map = np.append(idx_map, np.where(atom_types == ii))
    return idx_map


# TODO not really sure if the docstring is right the purpose of this is a bit unclear
def make_default_mesh(
    test_box: np.ndarray, cell_size: float = 3.0
) -> np.ndarray:
    """Get number of cells of size=`cell_size` fit into average box.

    Parameters
    ----------
    test_box : np.ndarray
        numpy array with cells of shape Nx9
    cell_size : float, optional
        length of one cell, by default 3.0

    Returns
    -------
    np.ndarray
        mesh for supplied boxes, how many cells fit in each direction
    """
    cell_lengths = np.linalg.norm(test_box.reshape([-1, 3, 3]), axis=2)
    avg_cell_lengths = np.average(cell_lengths, axis=0)
    ncell = (avg_cell_lengths / cell_size).astype(np.int32)
    ncell[ncell < 2] = 2
    default_mesh = np.zeros(6, dtype=np.int32)
    default_mesh[3:6] = ncell
    return default_mesh


# TODO not an ideal approach, every class uses this to parse arguments on its own, json
# TODO should be parsed once and the parsed result passed to all objects that need it
class ClassArg:
    """Class that take care of input json/yaml parsing.

    The rules for parsing are defined by the `add` method, than `parse` is called to
    process the supplied dict

    Attributes
    ----------
    arg_dict: Dict[str, Any]
        dictionary containing parsing rules
    alias_map: Dict[str, Any]
        dictionary with keyword aliases
    """

    def __init__(self) -> None:
        self.arg_dict = {}
        self.alias_map = {}

    def add(
        self,
        key: str,
        types_: Union[type, List[type]],
        alias: Optional[Union[str, List[str]]] = None,
        default: Any = None,
        must: bool = False,
    ) -> "ClassArg":
        """Add key to be parsed.

        Parameters
        ----------
        key : str
            key name
        types_ : Union[type, List[type]]
            list of allowed key types
        alias : Optional[Union[str, List[str]]], optional
            alias for the key, by default None
        default : Any, optional
            default value for the key, by default None
        must : bool, optional
            if the key is mandatory, by default False

        Returns
        -------
        ClassArg
            instance with added key
        """
        if not isinstance(types_, list):
            types = [types_]
        else:
            types = types_
        if alias is not None:
            if not isinstance(alias, list):
                alias_ = [alias]
            else:
                alias_ = alias
        else:
            alias_ = []

        self.arg_dict[key] = {
            "types": types,
            "alias": alias_,
            "value": default,
            "must": must,
        }
        for ii in alias_:
            self.alias_map[ii] = key

        return self

    def _add_single(self, key: str, data: Any):
        vtype = type(data)
        if data is None:
            return data
        if not (vtype in self.arg_dict[key]["types"]):
            for tp in self.arg_dict[key]["types"]:
                try:
                    vv = tp(data)
                except TypeError:
                    pass
                else:
                    break
            else:
                raise TypeError(
                    f"cannot convert provided key {key} to type(s) "
                    f'{self.arg_dict[key]["types"]} '
                )
        else:
            vv = data
        self.arg_dict[key]["value"] = vv

    def _check_must(self):
        for kk in self.arg_dict:
            if self.arg_dict[kk]["must"] and self.arg_dict[kk]["value"] is None:
                raise RuntimeError(f"key {kk} must be provided")

    def parse(self, jdata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse input dictionary, use the rules defined by add method.

        Parameters
        ----------
        jdata : Dict[str, Any]
            loaded json/yaml data

        Returns
        -------
        Dict[str, Any]
            parsed dictionary
        """
        for kk in jdata.keys():
            if kk in self.arg_dict:
                key = kk
                self._add_single(key, jdata[kk])
            else:
                if kk in self.alias_map:
                    key = self.alias_map[kk]
                    self._add_single(key, jdata[kk])
        self._check_must()
        return self.get_dict()

    def get_dict(self) -> Dict[str, Any]:
        """Get dictionary built from rules defined by add method.

        Returns
        -------
        Dict[str, Any]
            settings dictionary with default values
        """
        ret = {}
        for kk in self.arg_dict.keys():
            ret[kk] = self.arg_dict[kk]["value"]
        return ret


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


def get_activation_func(
    activation_fn: "_ACTIVATION",
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Get activation function callable based on string name.

    Parameters
    ----------
    activation_fn : _ACTIVATION
        one of the defined activation functions

    Returns
    -------
    Callable[[tf.Tensor], tf.Tensor]
        correspondingg TF callable

    Raises
    ------
    RuntimeError
        if unknown activation function is specified
    """
    if activation_fn not in ACTIVATION_FN_DICT:
        raise RuntimeError(f"{activation_fn} is not a valid activation function")
    return ACTIVATION_FN_DICT[activation_fn]


def get_precision(precision: "_PRECISION") -> Any:
    """Convert str to TF DType constant.

    Parameters
    ----------
    precision : _PRECISION
        one of the allowed precisions

    Returns
    -------
    tf.python.framework.dtypes.DType
        appropriate TF constant

    Raises
    ------
    RuntimeError
        if supplied precision string does not have acorresponding TF constant
    """
    if precision not in PRECISION_DICT:
        raise RuntimeError(f"{precision} is not a valid precision")
    return PRECISION_DICT[precision]


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
    matches = [str(d) for d in Path(root_dir).rglob("*") if (d / "type.raw").is_file()]
    if (Path(root_dir) / "type.raw").is_file():
        matches += [root_dir]
    return matches


def docstring_parameter(*sub: Tuple[str, ...]):
    """Add parameters to object docstring.

    Parameters
    ----------
    sub: Tuple[str, ...]
        list of strings that will be inserted into prepared locations in docstring.

    Note
    ----
    Can be used on both object and classes.
    """

    @wraps
    def dec(obj: "_OBJ") -> "_OBJ":
        if obj.__doc__ is not None:
            obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


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
