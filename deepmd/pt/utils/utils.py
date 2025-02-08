# SPDX-License-Identifier: LGPL-3.0-or-later
import os
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

from .env import (
    DEVICE,
)
from .env import PRECISION_DICT as PT_PRECISION_DICT


class CustomSiluJit(torch.nn.Module):
    def __init__(self, threshold=3.0):
        super().__init__()
        self.threshold = threshold

        # Precompute parameters for the tanh replacement
        sigmoid_threshold = 1 / (1 + np.exp(-threshold))
        self.slope = float(
            sigmoid_threshold + threshold * sigmoid_threshold * (1 - sigmoid_threshold)
        )
        self.const = float(threshold * sigmoid_threshold)

        # Generate and compile Jiterator kernels
        self._generate_jiterator_code()
        self._compile_jiterator_kernels()
        self._define_autograd_functions()

    def _generate_jiterator_code(self):
        # Forward kernel
        self.forward_code = f"""
        template <typename T>
        T custom_silu_forward(T x) {{
            const T threshold = {self.threshold};
            const T slope = {self.slope};
            const T const_val = {self.const};

            T sig = 1.0 / (1.0 + exp(-x));
            T silu = x * sig;
            T tanh_part = tanh(slope * (x - threshold)) + const_val;

            return (x > threshold) ? tanh_part : silu;
        }}
        """

        # First-order gradient kernel
        self.backward_code = f"""
        template <typename T>
        T custom_silu_backward(T x) {{
            const T threshold = {self.threshold};
            const T slope = {self.slope};

            T sig = 1.0 / (1.0 + exp(-x));
            T grad_silu = sig * (1 + x * (1 - sig));

            T tanh_term = tanh(slope * (x - threshold));
            T grad_tanh = slope * (1 - tanh_term * tanh_term);

            T grad = (x > threshold) ? grad_tanh : grad_silu;
            return grad;
        }}
        """

        # Corrected second-order gradient kernel (FIXED HERE)
        self.double_backward_code = f"""
        template <typename T>
        T custom_silu_double_backward(T x, T grad_grad_output, T grad_output) {{
            const T threshold = {self.threshold};
            const T slope = {self.slope};

            T grad_grad;
            if (x > threshold) {{
                T tanh_term = tanh(slope * (x - threshold));
                grad_grad = -2 * slope * slope * tanh_term * (1 - tanh_term * tanh_term);
            }} else {{
                T sig = 1.0 / (1.0 + exp(-x));
                T sig_prime = sig * (1 - sig);
                grad_grad = sig_prime * (2 + x * (1 - 2 * sig));  // FIXED COEFFICIENT
            }}
            return grad_output * grad_grad * grad_grad_output;
        }}
        """

    def _compile_jiterator_kernels(self):
        self.jitted_forward = torch.cuda.jiterator._create_jit_fn(self.forward_code)
        self.jitted_backward = torch.cuda.jiterator._create_jit_fn(self.backward_code)
        self.jitted_double_backward = torch.cuda.jiterator._create_jit_fn(
            self.double_backward_code
        )

    def _define_autograd_functions(self):
        class CustomSiluForward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return self.jitted_forward(x)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return CustomSiluBackward.apply(x, grad_output)

        class CustomSiluBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, grad_output):
                grad = self.jitted_backward(x)
                ctx.save_for_backward(x, grad_output, grad)
                return grad * grad_output

            @staticmethod
            def backward(ctx, grad_grad_output):
                (x, grad_output, grad) = ctx.saved_tensors
                return self.jitted_double_backward(
                    x, grad_grad_output, grad_output
                ), grad * grad_grad_output

        self.CustomSiluForward = CustomSiluForward

    def forward(self, x):
        return self.CustomSiluForward.apply(x)


class CustomSiluOp(torch.nn.Module):
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

        if not hasattr(torch.ops.deepmd, "thsilu"):

            def thsilu(
                argument0: torch.Tensor,
                argument1: float,
                argument2: float,
                argument3: float,
            ) -> list[torch.Tensor]:
                raise NotImplementedError(
                    "thsilu is not available since customized PyTorch OP library is not built when freezing the model. "
                    "See documentation for model compression for details."
                )

            # Note: this hack cannot actually save a model that can be runned using LAMMPS.
            torch.ops.deepmd.thsilu = thsilu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.ops.deepmd.thsilu(
            x.contiguous(), self.slope, self.threshold, self.const
        )
        return result


class CustomSilu(torch.nn.Module):
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

        if not hasattr(torch.ops.deepmd, "thsilu"):

            def thsilu(
                argument0: torch.Tensor,
                argument1: float,
                argument2: float,
                argument3: float,
            ) -> list[torch.Tensor]:
                raise NotImplementedError(
                    "thsilu is not available since customized PyTorch OP library is not built when freezing the model. "
                    "See documentation for model compression for details."
                )

            # Note: this hack cannot actually save a model that can be runned using LAMMPS.
            torch.ops.deepmd.thsilu = thsilu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_part = F.silu(x)
        mask = x > self.threshold
        if torch.any(mask):
            tanh_part = torch.tanh(self.slope * (x - self.threshold)) + self.const
            return torch.where(x < self.threshold, silu_part, tanh_part)
        else:
            return silu_part


class CustomDSilu(torch.nn.Module):
    def __init__(self, threshold=3.0, sig_s=0.0):
        super().__init__()
        self.threshold = threshold
        self.sig_s = sig_s
        self.ex_threshold = float(np.exp(threshold - 1))
        self.ex_threshold_shift = float(np.exp(threshold - 1 + sig_s))

    def forward(self, x):
        exp_mx = torch.exp(-x)
        silu_x = x * (1 / (1 + exp_mx))
        exp_mx_threshold = exp_mx * self.ex_threshold
        silu_x_threshold = (x - (self.threshold - 1)) * (1 / (1 + exp_mx_threshold))
        exp_mx_threshold_shift = exp_mx * self.ex_threshold_shift
        sig_threshold_shift = 1 / (1 + exp_mx_threshold_shift)
        result = silu_x + sig_threshold_shift * (1 - silu_x_threshold)
        return result


class CustomDSiluOp(torch.nn.Module):
    def __init__(self, threshold=3.0, sig_s=3.0):
        super().__init__()
        self.threshold = threshold
        self.sig_s = sig_s

    def forward(self, x):
        result = torch.ops.deepmd.cdsilu(x.contiguous(), self.threshold, self.sig_s)
        return result


class ActivationFn(torch.nn.Module):
    def __init__(self, activation: Optional[str]) -> None:
        super().__init__()
        self.activation: str = activation if activation is not None else "linear"
        if self.activation.startswith("custom_silu"):
            threshold = (
                float(self.activation.split(":")[-1]) if ":" in self.activation else 3.0
            )
            # get op method from environment
            SILU_OP = os.environ.get("SILU_OP", "default")
            if SILU_OP == "default":
                self.custom_silu = CustomSilu(threshold=threshold)
            elif SILU_OP == "op":
                self.custom_silu = CustomSiluOp(threshold=threshold)
            elif SILU_OP == "jit":
                self.custom_silu = CustomSiluJit(threshold=threshold)
            else:
                raise ValueError(f"Not defined SILU_OP: {SILU_OP}!")
        else:
            self.custom_silu = None

        if self.activation.startswith("custom_dsilu"):
            threshold = (
                float(self.activation.split(":")[-1]) if ":" in self.activation else 3.0
            )
            SILU_OP = os.environ.get("SILU_OP", "default")
            if SILU_OP == "default":
                self.custom_dsilu = CustomDSilu(threshold=threshold)
            elif SILU_OP == "op":
                self.custom_dsilu = CustomDSiluOp(threshold=threshold)
            else:
                raise ValueError(f"Not defined SILU_OP: {SILU_OP}!")
        else:
            self.custom_dsilu = None

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
        elif self.activation.startswith("custom_silu"):
            assert self.custom_silu is not None
            return self.custom_silu(x)
        elif self.activation.startswith("custom_dsilu"):
            assert self.custom_dsilu is not None
            return self.custom_dsilu(x)
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
