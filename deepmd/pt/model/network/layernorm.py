# SPDX-License-Identifier: LGPL-3.0-or-later
import torch
import torch.nn as nn

from deepmd.dpmodel.utils.network import LayerNorm as DPLayerNorm
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from .mlp import (
    MLPLayer,
)

device = env.DEVICE


class LayerNorm(MLPLayer):
    def __init__(
        self,
        num_in,
        eps: float = 1e-5,
        uni_init: bool = True,
        bavg: float = 0.0,
        stddev: float = 1.0,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ):
        self.eps = eps
        self.uni_init = uni_init
        self.num_in = num_in
        super().__init__(
            num_in=1,
            num_out=num_in,
            bias=True,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            bavg=bavg,
            stddev=stddev,
            precision=precision,
        )
        self.matrix = torch.nn.Parameter(self.matrix.squeeze(0))
        if self.uni_init:
            nn.init.ones_(self.matrix.data)
            nn.init.zeros_(self.bias.data)
        self.trainable = trainable
        if not self.trainable:
            self.matrix.requires_grad = False
            self.bias.requires_grad = False

    def dim_out(self) -> int:
        return self.matrix.shape[0]

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """One Layer Norm used by DP model.

        Parameters
        ----------
        xx : torch.Tensor
            The input of index.

        Returns
        -------
        yy: torch.Tensor
            The output.
        """
        mean = xx.mean(dim=-1, keepdim=True)
        variance = xx.var(dim=-1, unbiased=False, keepdim=True)
        yy = (xx - mean) / torch.sqrt(variance + self.eps)
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
            return (
                nn.Parameter(data=to_torch_tensor(nl[ss]))
                if nl[ss] is not None
                else None
            )

        obj.matrix = check_load_param("matrix")
        obj.bias = check_load_param("bias")
        return obj
