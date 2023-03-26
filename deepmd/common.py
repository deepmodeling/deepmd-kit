"""Collection of functions and classes used throughout the whole package."""

import json
import warnings
from functools import (
    wraps,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import tensorflow
import yaml
from tensorflow.python.framework import (
    tensor_util,
)

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    op_module,
    tf,
)
from deepmd.utils.path import (
    DPPath,
)

if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    _OBJ = TypeVar("_OBJ")
    try:
        from typing import Literal  # python >3.6
    except ImportError:
        from typing_extensions import Literal  # type: ignore
    _ACTIVATION = Literal[
        "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu", "gelu_tf"
    ]
    _PRECISION = Literal["default", "float16", "float32", "float64"]

# define constants
PRECISION_DICT = {
    "default": GLOBAL_TF_FLOAT_PRECISION,
    "float16": tf.float16,
    "float32": tf.float32,
    "float64": tf.float64,
    "bfloat16": tf.bfloat16,
}


def gelu(x: tf.Tensor) -> tf.Tensor:
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU, implemented by custom operator.

    Parameters
    ----------
    x : tf.Tensor
        float Tensor to perform activation

    Returns
    -------
    tf.Tensor
        `x` with the GELU activation applied

    References
    ----------
    Original paper
    https://arxiv.org/abs/1606.08415
    """
    return op_module.gelu_custom(x)


def gelu_tf(x: tf.Tensor) -> tf.Tensor:
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU, implemented by TF.

    Parameters
    ----------
    x : tf.Tensor
        float Tensor to perform activation

    Returns
    -------
    tf.Tensor
        `x` with the GELU activation applied

    References
    ----------
    Original paper
    https://arxiv.org/abs/1606.08415
    """

    def gelu_wrapper(x):
        try:
            return tensorflow.nn.gelu(x, approximate=True)
        except AttributeError:
            warnings.warn(
                "TensorFlow does not provide an implementation of gelu, please upgrade your TensorFlow version. Fallback to the custom gelu operator."
            )
            return op_module.gelu_custom(x)

    return (lambda x: gelu_wrapper(x))(x)


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
    "gelu_tf": gelu_tf,
    "None": None,
    "none": None,
}


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


# TODO not really sure if the docstring is right the purpose of this is a bit unclear
def make_default_mesh(test_box: np.ndarray, cell_size: float = 3.0) -> np.ndarray:
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
    activation_fn: Union["_ACTIVATION", None],
) -> Union[Callable[[tf.Tensor], tf.Tensor], None]:
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
    if activation_fn is None:
        return None
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


def safe_cast_tensor(
    input: tf.Tensor, from_precision: tf.DType, to_precision: tf.DType
) -> tf.Tensor:
    """Convert a Tensor from a precision to another precision.

    If input is not a Tensor or without the specific precision, the method will not
    cast it.

    Parameters
    ----------
    input : tf.Tensor
        input tensor
    from_precision : tf.DType
        Tensor data type that is casted from
    to_precision : tf.DType
        Tensor data type that casts to

    Returns
    -------
    tf.Tensor
        casted Tensor
    """
    if tensor_util.is_tensor(input) and input.dtype == from_precision:
        return tf.cast(input, to_precision)
    return input


def cast_precision(func: Callable) -> Callable:
    """A decorator that casts and casts back the input
    and output tensor of a method.

    The decorator should be used in a classmethod.

    The decorator will do the following thing:
    (1) It casts input Tensors from `GLOBAL_TF_FLOAT_PRECISION`
    to precision defined by property `precision`.
    (2) It casts output Tensors from `precision` to
    `GLOBAL_TF_FLOAT_PRECISION`.
    (3) It checks inputs and outputs and only casts when
    input or output is a Tensor and its dtype matches
    `GLOBAL_TF_FLOAT_PRECISION` and `precision`, respectively.
    If it does not match (e.g. it is an integer), the decorator
    will do nothing on it.

    Returns
    -------
    Callable
        a decorator that casts and casts back the input and
        output tensor of a method

    Examples
    --------
    >>> class A:
    ...   @property
    ...   def precision(self):
    ...     return tf.float32
    ...
    ...   @cast_precision
    ...   def f(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    ...     return x ** 2 + y
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # only convert tensors
        returned_tensor = func(
            self,
            *[
                safe_cast_tensor(vv, GLOBAL_TF_FLOAT_PRECISION, self.precision)
                for vv in args
            ],
            **{
                kk: safe_cast_tensor(vv, GLOBAL_TF_FLOAT_PRECISION, self.precision)
                for kk, vv in kwargs.items()
            },
        )
        if isinstance(returned_tensor, tuple):
            return tuple(
                safe_cast_tensor(vv, self.precision, GLOBAL_TF_FLOAT_PRECISION)
                for vv in returned_tensor
            )
        else:
            return safe_cast_tensor(
                returned_tensor, self.precision, GLOBAL_TF_FLOAT_PRECISION
            )

    return wrapper


def clear_session():
    """Reset all state generated by DeePMD-kit."""
    tf.reset_default_graph()
    # TODO: remove this line when data_requirement is not a global variable
    data_requirement.clear()
