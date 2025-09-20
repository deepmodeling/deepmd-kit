# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
    overload,
)

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from deepmd.dpmodel.common import PRECISION_DICT as NP_PRECISION_DICT
from deepmd.pt.utils import (
    env,
)

from .env import (
    DEVICE,
)
from .env import PRECISION_DICT as PT_PRECISION_DICT


def silut_forward(
    x: torch.Tensor, threshold: float, slope: float, const_val: float
) -> torch.Tensor:
    sig = torch.sigmoid(x)
    silu = x * sig
    tanh = torch.tanh(slope * (x - threshold)) + const_val
    return torch.where(x >= threshold, tanh, silu)


def silut_backward(
    x: torch.Tensor, grad_output: torch.Tensor, threshold: float, slope: float
) -> torch.Tensor:
    sig = torch.sigmoid(x)
    grad_silu = sig * (1 + x * (1 - sig))

    tanh = torch.tanh(slope * (x - threshold))
    grad_tanh = slope * (1 - tanh * tanh)

    grad = torch.where(x >= threshold, grad_tanh, grad_silu)
    return grad * grad_output


def silut_double_backward(
    x: torch.Tensor,
    grad_grad_output: torch.Tensor,
    grad_output: torch.Tensor,
    threshold: float,
    slope: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SiLU branch
    sig = torch.sigmoid(x)

    sig_prime = sig * (1 - sig)
    grad_silu = sig + x * sig_prime
    grad_grad_silu = sig_prime * (2 + x * (1 - 2 * sig))

    # Tanh branch
    tanh = torch.tanh(slope * (x - threshold))
    tanh_square = tanh * tanh  #  .square is slow for jit.script!
    grad_tanh = slope * (1 - tanh_square)
    grad_grad_tanh = -2 * slope * tanh * grad_tanh

    grad = torch.where(x >= threshold, grad_tanh, grad_silu)
    grad_grad = torch.where(x >= threshold, grad_grad_tanh, grad_grad_silu)
    return grad_output * grad_grad * grad_grad_output, grad * grad_grad_output


class SiLUTScript(torch.nn.Module):
    def __init__(self, threshold: float = 3.0):
        super().__init__()
        self.threshold = threshold

        # Precompute parameters for the tanh replacement
        sigmoid_threshold = 1 / (1 + np.exp(-threshold))
        self.slope = float(
            sigmoid_threshold + threshold * sigmoid_threshold * (1 - sigmoid_threshold)
        )
        self.const_val = float(threshold * sigmoid_threshold)
        self.get_script_code()

    def get_script_code(self):
        silut_forward_script = torch.jit.script(silut_forward)
        silut_backward_script = torch.jit.script(silut_backward)
        silut_double_backward_script = torch.jit.script(silut_double_backward)

        class SiLUTFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, threshold, slope, const_val):
                ctx.save_for_backward(x)
                ctx.threshold = threshold
                ctx.slope = slope
                ctx.const_val = const_val
                return silut_forward_script(x, threshold, slope, const_val)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                threshold = ctx.threshold
                slope = ctx.slope

                grad_input = SiLUTGradFunction.apply(x, grad_output, threshold, slope)
                return grad_input, None, None, None

        class SiLUTGradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, grad_output, threshold, slope):
                ctx.threshold = threshold
                ctx.slope = slope
                grad_input = silut_backward_script(x, grad_output, threshold, slope)
                ctx.save_for_backward(x, grad_output)
                return grad_input

            @staticmethod
            def backward(ctx, grad_grad_output):
                (x, grad_output) = ctx.saved_tensors
                threshold = ctx.threshold
                slope = ctx.slope

                grad_input, grad_mul_grad_grad_output = silut_double_backward_script(
                    x, grad_grad_output, grad_output, threshold, slope
                )
                return grad_input, grad_mul_grad_grad_output, None, None

        self.SiLUTFunction = SiLUTFunction

    def forward(self, x):
        return self.SiLUTFunction.apply(x, self.threshold, self.slope, self.const_val)


class SiLUT(torch.nn.Module):
    def __init__(self, threshold=3.0):
        super().__init__()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def silu(x):
            return x * sigmoid(x)

        def silu_grad(x):
            sig = sigmoid(x)
            return sig + x * sig * (1 - sig)

        self.threshold = threshold
        self.slope = float(silu_grad(threshold))
        self.const = float(silu(threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        silu = x * sig
        tanh = torch.tanh(self.slope * (x - self.threshold)) + self.const
        return torch.where(x >= self.threshold, tanh, silu)


class ActivationFn(torch.nn.Module):
    def __init__(self, activation: Optional[str]) -> None:
        super().__init__()
        self.activation: str = activation if activation is not None else "linear"
        if self.activation.lower().startswith(
            "silut"
        ) or self.activation.lower().startswith("custom_silu"):
            threshold = (
                float(self.activation.split(":")[-1]) if ":" in self.activation else 3.0
            )
            if env.CUSTOM_OP_USE_JIT:
                # for efficient training but can not be jit
                self.silut = SiLUTScript(threshold=threshold)
            else:
                # for jit freeze
                self.silut = SiLUT(threshold=threshold)
        else:
            self.silut = None

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
        elif self.activation.lower() == "silu":
            return F.silu(x)
        elif self.activation.lower().startswith(
            "silut"
        ) or self.activation.lower().startswith("custom_silu"):
            assert self.silut is not None
            return self.silut(x)
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


def dict_to_device(sample_dict) -> None:
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


def hashmix(value: int, hash_const: list[int]):
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


def mix_entropy(entropy_array: list[int]) -> int:
    # https://github.com/numpy/numpy/blob/a4cddb60489f821a1a4dffc16cd5c69755d43bdb/numpy/random/bit_generator.pyx#L341-L374
    hash_const = [INIT_A]
    mixer = hashmix(entropy_array[0], hash_const)
    for i_src in range(1, len(entropy_array)):
        mixer = mix(mixer, hashmix(entropy_array[i_src], hash_const))
    return mixer


def get_generator(
    seed: Optional[Union[int, list[int]]] = None,
) -> Optional[torch.Generator]:
    if seed is not None:
        if isinstance(seed, list):
            seed = mix_entropy(seed)
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)
        return generator
    else:
        return None
