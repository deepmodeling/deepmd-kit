# SPDX-License-Identifier: LGPL-3.0-or-later
"""Collection of functions and classes used throughout the whole package."""

import warnings
from functools import (
    wraps,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Union,
)

import tensorflow
from tensorflow.python.framework import (
    tensor_util,
)

from deepmd.common import (
    add_data_requirement,
    data_requirement,
    expand_sys_str,
    get_np_precision,
    j_loader,
    j_must_have,
    make_default_mesh,
    select_idx_map,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    op_module,
    tf,
)

if TYPE_CHECKING:
    from deepmd.common import (
        _ACTIVATION,
        _PRECISION,
    )

__all__ = [
    # from deepmd.common
    "data_requirement",
    "add_data_requirement",
    "select_idx_map",
    "make_default_mesh",
    "j_must_have",
    "j_loader",
    "expand_sys_str",
    "get_np_precision",
    # from self
    "PRECISION_DICT",
    "gelu",
    "gelu_tf",
    "ACTIVATION_FN_DICT",
    "get_activation_func",
    "get_precision",
    "safe_cast_tensor",
    "cast_precision",
    "clear_session",
]

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
    ...     @property
    ...     def precision(self):
    ...         return tf.float32
    ...
    ...     @cast_precision
    ...     def f(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    ...         return x**2 + y
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
