# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Union,
    overload,
)

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from deepmd.dpmodel.common import PRECISION_DICT as NP_PRECISION_DICT

from .env import (
    DEVICE,
)
from .env import PRECISION_DICT as PT_PRECISION_DICT


class ActivationFn(torch.nn.Module):
    def __init__(self, activation: Optional[str]):
        super().__init__()
        self.activation: str = activation if activation is not None else "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the tensor after applying activation function corresponding to `activation`."""
        # See jit supported types: https://pytorch.org/docs/stable/jit_language_reference.html#supported-type

        if self.activation.lower() == "relu":
            return F.relu(x)
        elif self.activation.lower() == "gelu" or self.activation.lower() == "gelu_tf":
            return F.gelu(x, approximate="tanh")
        elif self.activation.lower() == "tanh":
            return torch.tanh(x)
        elif self.activation.lower() == "relu6":
            return F.relu6(x)
        elif self.activation.lower() == "softplus":
            return F.softplus(x)
        elif self.activation.lower() == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation.lower() == "linear" or self.activation.lower() == "none":
            return x
        else:
            raise RuntimeError(f"activation function {self.activation} not supported")


@overload
def to_numpy_array(xx: torch.Tensor) -> np.ndarray: ...


@overload
def to_numpy_array(xx: None) -> None: ...


def to_numpy_array(
    xx,
):
    if xx is None:
        return None
    assert xx is not None
    # Create a reverse mapping of PT_PRECISION_DICT
    reverse_precision_dict = {v: k for k, v in PT_PRECISION_DICT.items()}
    # Use the reverse mapping to find keys with the desired value
    prec = reverse_precision_dict.get(xx.dtype, None)
    prec = NP_PRECISION_DICT.get(prec, None)
    if prec is None:
        raise ValueError(f"unknown precision {xx.dtype}")
    if xx.dtype == torch.bfloat16:
        # https://github.com/pytorch/pytorch/issues/109873
        xx = xx.float()
    return xx.detach().cpu().numpy().astype(prec)


@overload
def to_torch_tensor(xx: np.ndarray) -> torch.Tensor: ...


@overload
def to_torch_tensor(xx: None) -> None: ...


def to_torch_tensor(
    xx,
):
    if xx is None:
        return None
    assert xx is not None
    if not isinstance(xx, np.ndarray):
        return xx
    # Create a reverse mapping of NP_PRECISION_DICT
    reverse_precision_dict = {v: k for k, v in NP_PRECISION_DICT.items()}
    # Use the reverse mapping to find keys with the desired value
    prec = reverse_precision_dict.get(xx.dtype.type, None)
    prec = PT_PRECISION_DICT.get(prec, None)
    if prec is None:
        raise ValueError(f"unknown precision {xx.dtype}")
    if xx.dtype == ml_dtypes.bfloat16:
        # https://github.com/pytorch/pytorch/issues/109873
        xx = xx.astype(np.float32)
    return torch.tensor(xx, dtype=prec, device=DEVICE)


def dict_to_device(sample_dict):
    for key in sample_dict:
        if isinstance(sample_dict[key], list):
            sample_dict[key] = [item.to(DEVICE) for item in sample_dict[key]]
        if isinstance(sample_dict[key], np.float32):
            sample_dict[key] = (
                torch.ones(1, dtype=torch.float32, device=DEVICE) * sample_dict[key]
            )
        else:
            if sample_dict[key] is not None:
                sample_dict[key] = sample_dict[key].to(DEVICE)


# https://github.com/numpy/numpy/blob/a4cddb60489f821a1a4dffc16cd5c69755d43bdb/numpy/random/bit_generator.pyx#L58-L63
INIT_A = 0x43B0D7E5
MULT_A = 0x931E8875
MIX_MULT_L = 0xCA01F9DD
MIX_MULT_R = 0x4973F715
XSHIFT = 16


def hashmix(value: int, hash_const: List[int]):
    value ^= INIT_A
    hash_const[0] *= MULT_A
    value *= INIT_A
    # prevent overflow
    hash_const[0] &= 0xFFFF_FFFF_FFFF_FFFF
    value &= 0xFFFF_FFFF_FFFF_FFFF
    value ^= value >> XSHIFT
    return value


def mix(x: int, y: int):
    result = MIX_MULT_L * x - MIX_MULT_R * y
    # prevent overflow
    result &= 0xFFFF_FFFF_FFFF_FFFF
    result ^= result >> XSHIFT
    return result


def mix_entropy(entropy_array: List[int]) -> int:
    # https://github.com/numpy/numpy/blob/a4cddb60489f821a1a4dffc16cd5c69755d43bdb/numpy/random/bit_generator.pyx#L341-L374
    hash_const = [INIT_A]
    mixer = hashmix(entropy_array[0], hash_const)
    for i_src in range(1, len(entropy_array)):
        mixer = mix(mixer, hashmix(entropy_array[i_src], hash_const))
    return mixer


def get_generator(
    seed: Optional[Union[int, List[int]]] = None,
) -> Optional[torch.Generator]:
    if seed is not None:
        if isinstance(seed, list):
            seed = mix_entropy(seed)
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)
        return generator
    else:
        return None
