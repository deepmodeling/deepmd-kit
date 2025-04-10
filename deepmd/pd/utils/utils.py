# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    overload,
)

import ml_dtypes
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.framework import (
    core,
)

from deepmd.dpmodel.common import PRECISION_DICT as NP_PRECISION_DICT

from .env import (
    DEVICE,
)
from .env import PRECISION_DICT as PD_PRECISION_DICT

if TYPE_CHECKING:
    from deepmd.pd.model.network.init import (
        PaddleGenerator,
    )


class SiLUT(paddle.nn.Layer):
    def __init__(self, threshold=3.0):
        super().__init__()

        def sigmoid(x):
            return paddle.nn.functional.sigmoid(x)

        def silu(x):
            return paddle.nn.functional.silu(x)

        def silu_grad(x):
            sig = sigmoid(x)
            return sig + x * sig * (1 - sig)

        self.threshold = paddle.to_tensor(threshold)
        self.slope = float(silu_grad(self.threshold))
        self.const = float(silu(self.threshold))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        silu_part = F.silu(x)

        # NOTE: control flow to be fixed in to_static
        # mask = x >= self.threshold
        # if paddle.any(mask).item():
        #     tanh_part = paddle.tanh(self.slope * (x - self.threshold)) + self.const
        #     return paddle.where(x < self.threshold, silu_part, tanh_part)
        # else:
        #     return silu_part

        # NOTE: workaround
        tanh_part = paddle.tanh(self.slope * (x - self.threshold)) + self.const
        return paddle.where(x < self.threshold, silu_part, tanh_part)


class ActivationFn(paddle.nn.Layer):
    def __init__(self, activation: str | None):
        super().__init__()
        self.activation: str = activation if activation is not None else "linear"
        if self.activation.lower().startswith(
            "silut"
        ) or self.activation.lower().startswith("custom_silu"):
            threshold = (
                float(self.activation.split(":")[-1]) if ":" in self.activation else 3.0
            )
            self.silut = SiLUT(threshold=threshold)
        else:
            self.silut = None

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Returns the tensor after applying activation function corresponding to `activation`."""
        if self.activation.lower() == "relu":
            return F.relu(x)
        elif self.activation.lower() == "gelu" or self.activation.lower() == "gelu_tf":
            return F.gelu(x, approximate=True)
        elif self.activation.lower() == "tanh":
            return paddle.tanh(x)
        elif self.activation.lower() == "relu6":
            return F.relu6(x)
        elif self.activation.lower() == "softplus":
            return F.softplus(x)
        elif self.activation.lower() == "sigmoid":
            return F.sigmoid(x)
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
def to_numpy_array(xx: paddle.Tensor) -> np.ndarray: ...


@overload
def to_numpy_array(xx: None) -> None: ...


def to_numpy_array(
    xx,
):
    if xx is None:
        return None
    assert xx is not None
    # Create a reverse mapping of PD_PRECISION_DICT
    reverse_precision_dict = {v: k for k, v in PD_PRECISION_DICT.items()}
    # Use the reverse mapping to find keys with the desired value
    prec = reverse_precision_dict.get(xx.dtype, None)
    prec = NP_PRECISION_DICT.get(prec, np.float64)
    if prec is None:
        raise ValueError(f"unknown precision {xx.dtype}")
    if isinstance(xx, np.ndarray):
        return xx.astype(prec)
    if xx.dtype == paddle.bfloat16:
        xx = xx.astype(paddle.get_default_dtype())
    return xx.numpy().astype(prec)


@overload
def to_paddle_tensor(xx: np.ndarray) -> paddle.Tensor: ...


@overload
def to_paddle_tensor(xx: None) -> None: ...


def to_paddle_tensor(
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
    prec = PD_PRECISION_DICT.get(prec, None)
    if prec is None:
        raise ValueError(f"unknown precision {xx.dtype}")
    if xx.dtype == ml_dtypes.bfloat16:
        xx = xx.astype(np.float32)
    return paddle.to_tensor(xx, dtype=prec, place=DEVICE)


def dict_to_device(sample_dict):
    for key in sample_dict:
        if isinstance(sample_dict[key], list):
            sample_dict[key] = [item.to(DEVICE) for item in sample_dict[key]]
        if isinstance(sample_dict[key], np.float32):
            sample_dict[key] = (
                paddle.ones(1, dtype=paddle.float32).to(device=DEVICE)
                * sample_dict[key]
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
    seed: int | list[int] | None = None,
) -> PaddleGenerator | None:
    if seed is not None:
        if isinstance(seed, list):
            seed = mix_entropy(seed)
        if DEVICE == "cpu":
            generator = paddle.framework.core.default_cpu_generator()
        elif DEVICE == "gpu":
            generator = paddle.framework.core.default_cuda_generator(0)
        elif DEVICE.startswith("gpu:"):
            generator = paddle.framework.core.default_cuda_generator(
                int(DEVICE.split("gpu:")[1])
            )
        else:
            raise ValueError("DEVICE should be cpu or gpu or gpu:x")
        generator.manual_seed(seed)
        return generator
    else:
        return None


@contextmanager
def nvprof_context(enable_profiler: bool, name: str):
    if enable_profiler:
        core.nvprof_nvtx_push(name)

    try:
        yield

    finally:
        if enable_profiler:
            core.nvprof_nvtx_pop()
