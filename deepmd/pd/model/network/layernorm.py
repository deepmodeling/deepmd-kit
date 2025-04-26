# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import numpy as np
import paddle
import paddle.nn as nn

from deepmd.dpmodel.utils.network import LayerNorm as DPLayerNorm
from deepmd.pd.model.network.init import (
    normal_,
    ones_,
    zeros_,
)
from deepmd.pd.utils import (
    decomp,
    env,
)
from deepmd.pd.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pd.utils.utils import (
    get_generator,
    to_numpy_array,
    to_paddle_tensor,
)

device = env.DEVICE


def empty_t(shape, precision):
    return paddle.empty(shape, dtype=precision).to(device=device)


class LayerNorm(nn.Layer):
    def __init__(
        self,
        num_in,
        eps: float = 1e-5,
        uni_init: bool = True,
        bavg: float = 0.0,
        stddev: float = 1.0,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
    ):
        super().__init__()
        self.eps = eps
        self.uni_init = uni_init
        self.num_in = num_in
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.matrix = self.create_parameter(
            shape=[num_in],
            dtype=self.prec,
            default_initializer=nn.initializer.Assign(
                empty_t([num_in], self.prec),
            ),
        )
        self.bias = self.create_parameter(
            shape=[num_in],
            dtype=self.prec,
            default_initializer=nn.initializer.Assign(empty_t([num_in], self.prec)),
        )
        random_generator = get_generator(seed)
        if self.uni_init:
            ones_(self.matrix.data)
            zeros_(self.bias.data)
        else:
            normal_(self.bias.data, mean=bavg, std=stddev, generator=random_generator)
            normal_(
                self.matrix.data,
                std=stddev / np.sqrt(self.num_in),
                generator=random_generator,
            )
        self.trainable = trainable
        if not self.trainable:
            self.matrix.stop_gradient = True
            self.bias.stop_gradient = True

    def dim_out(self) -> int:
        return self.matrix.shape[0]

    def forward(
        self,
        xx: paddle.Tensor,
    ) -> paddle.Tensor:
        """One Layer Norm used by DP model.

        Parameters
        ----------
        xx : paddle.Tensor
            The input of index.

        Returns
        -------
        yy: paddle.Tensor
            The output.
        """
        # NOTE: control flow with double backward is not supported well yet by paddle.jit
        if not paddle.in_dynamic_mode() or decomp.numel(xx) > 0:
            variance, mean = (
                paddle.var(xx, axis=-1, unbiased=False, keepdim=True),
                paddle.mean(xx, axis=-1, keepdim=True),
            )
            yy = (xx - mean) / paddle.sqrt(variance + self.eps)
        else:
            yy = xx
        if self.matrix is not None and self.bias is not None:
            yy = yy * self.matrix + self.bias
        return yy

    def serialize(self) -> dict:
        """Serialize the layer to a dict.

        Returns
        -------
        dict
            The serialized layer.
        """
        nl = DPLayerNorm(
            self.matrix.shape[0],
            eps=self.eps,
            trainable=self.trainable,
            precision=self.precision,
        )
        nl.w = to_numpy_array(self.matrix)
        nl.b = to_numpy_array(self.bias)
        data = nl.serialize()
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "LayerNorm":
        """Deserialize the layer from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        nl = DPLayerNorm.deserialize(data)
        obj = cls(
            nl["matrix"].shape[0],
            eps=nl["eps"],
            trainable=nl["trainable"],
            precision=nl["precision"],
        )
        prec = PRECISION_DICT[obj.precision]

        def check_load_param(ss):
            if nl[ss] is not None:
                tensor = to_paddle_tensor(nl[ss])
                return paddle.create_parameter(
                    tensor.shape,
                    dtype=tensor.dtype,
                    default_initializer=nn.initializer.Assign(tensor),
                )
            return None

        obj.matrix = check_load_param("matrix")
        obj.bias = check_load_param("bias")
        return obj
