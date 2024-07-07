from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Union,
)

from deepmd.common import (
    VALID_ACTIVATION,
    VALID_PRECISION,
    expand_sys_str,
    get_np_precision,
    j_loader,
    make_default_mesh,
    select_idx_map,
)

from deepmd.pt.env import (
    GLOBAL_PT_FLOAT_PRECISION,
    torch,
)

import torch.nn.functional as F

if TYPE_CHECKING:
    from deepmd.common import (
        _ACTIVATION,
        _PRECISION,
    )

ACTIVATION_FN_DICT = {
    "relu": F.relu,
    "relu6": F.relu6,
    "softplus": F.softplus,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    # PyTorch has no gelu_tf
    "gelu_tf": lambda x: x * 0.5 * (1.0 + torch.erf(x / 1.41421)),
    "linear": lambda x: x,
    "none": lambda x: x,
}
assert VALID_ACTIVATION.issubset(ACTIVATION_FN_DICT.keys())

PRECISION_DICT = {
    "default": GLOBAL_PT_FLOAT_PRECISION,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}
assert VALID_PRECISION.issubset(PRECISION_DICT.keys())

def get_activation_func(
    activation_fn: Union[_ACTIVATION, None]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation function callable based on string name.

    Parameters
    ----------
    activation_fn : _ACTIVATION
        One of the defined activation functions

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Corresponding PyTorch callable

    Raises
    ------
    RuntimeError
        If unknown activation function is specified
    """
    if activation_fn is None:
        activation_fn = "none"
    assert activation_fn is not None
    if activation_fn.lower() not in ACTIVATION_FN_DICT:
        raise RuntimeError(f"{activation_fn} is not a valid activation function")
    return ACTIVATION_FN_DICT[activation_fn.lower()]


def get_precision(precision: _PRECISION) -> Any:
    """Convert str to PyTorch dtype constant.

    Parameters
    ----------
    precision : _PRECISION
        One of the allowed precisions

    Returns
    -------
    torch.dtype
        Appropriate PyTorch dtype constant

    Raises
    ------
    RuntimeError
        If supplied precision string does not have a corresponding PyTorch dtype constant
    """
    if precision not in PRECISION_DICT:
        raise RuntimeError(f"{precision} is not a valid precision")
    return PRECISION_DICT[precision]