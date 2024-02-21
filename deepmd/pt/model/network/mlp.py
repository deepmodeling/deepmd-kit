# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
    Dict,
    Optional,
)

import numpy as np
import torch
import torch.nn as nn

from deepmd.pt.utils import (
    env,
)

device = env.DEVICE

from deepmd.dpmodel.utils import (
    NativeLayer,
)
from deepmd.dpmodel.utils import NetworkCollection as DPNetworkCollection
from deepmd.dpmodel.utils import (
    make_embedding_network,
    make_fitting_network,
    make_multilayer_network,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"


def empty_t(shape, precision):
    return torch.empty(shape, dtype=precision, device=device)


class MLPLayer(nn.Module):
    def __init__(
        self,
        num_in,
        num_out,
        bias: bool = True,
        use_timestep: bool = False,
        activation_function: Optional[str] = None,
        resnet: bool = False,
        bavg: float = 0.0,
        stddev: float = 1.0,
        precision: str = DEFAULT_PRECISION,
    ):
        super().__init__()
        # only use_timestep when skip connection is established.
        self.use_timestep = use_timestep and (
            num_out == num_in or num_out == num_in * 2
        )
        self.activate_name = activation_function
        self.activate = ActivationFn(self.activate_name)
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.matrix = nn.Parameter(data=empty_t((num_in, num_out), self.prec))
        nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        if bias:
            self.bias = nn.Parameter(
                data=empty_t([num_out], self.prec),
            )
            nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        else:
            self.bias = None
        if self.use_timestep:
            self.idt = nn.Parameter(data=empty_t([num_out], self.prec))
            nn.init.normal_(self.idt.data, mean=0.1, std=0.001)
        else:
            self.idt = None
        self.resnet = resnet

    def check_type_consistency(self):
        precision = self.precision

        def check_var(var):
            if var is not None:
                # assertion "float64" == "double" would fail
                assert PRECISION_DICT[var.dtype.name] is PRECISION_DICT[precision]

        check_var(self.w)
        check_var(self.b)
        check_var(self.idt)

    def dim_in(self) -> int:
        return self.matrix.shape[0]

    def dim_out(self) -> int:
        return self.matrix.shape[1]

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """One MLP layer used by DP model.

        Parameters
        ----------
        xx : torch.Tensor
            The input.

        Returns
        -------
        yy: torch.Tensor
            The output.
        """
        ori_prec = xx.dtype
        xx = xx.to(self.prec)
        yy = (
            torch.matmul(xx, self.matrix) + self.bias
            if self.bias is not None
            else torch.matmul(xx, self.matrix)
        )
        yy = self.activate(yy).clone()
        yy = yy * self.idt if self.idt is not None else yy
        if self.resnet:
            if xx.shape[-1] == yy.shape[-1]:
                yy += xx
            elif 2 * xx.shape[-1] == yy.shape[-1]:
                yy += torch.concat([xx, xx], dim=-1)
            else:
                yy = yy
        yy = yy.to(ori_prec)
        return yy

    def serialize(self) -> dict:
        """Serialize the layer to a dict.

        Returns
        -------
        dict
            The serialized layer.
        """
        nl = NativeLayer(
            self.matrix.shape[0],
            self.matrix.shape[1],
            bias=self.bias is not None,
            use_timestep=self.idt is not None,
            activation_function=self.activate_name,
            resnet=self.resnet,
            precision=self.precision,
        )
        nl.w, nl.b, nl.idt = (
            self.matrix.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy() if self.bias is not None else None,
            self.idt.detach().cpu().numpy() if self.idt is not None else None,
        )
        return nl.serialize()

    @classmethod
    def deserialize(cls, data: dict) -> "MLPLayer":
        """Deserialize the layer from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        nl = NativeLayer.deserialize(data)
        obj = cls(
            nl["matrix"].shape[0],
            nl["matrix"].shape[1],
            bias=nl["bias"] is not None,
            use_timestep=nl["idt"] is not None,
            activation_function=nl["activation_function"],
            resnet=nl["resnet"],
            precision=nl["precision"],
        )
        prec = PRECISION_DICT[obj.precision]

        def check_load_param(ss):
            return (
                nn.Parameter(data=torch.tensor(nl[ss], dtype=prec, device=device))
                if nl[ss] is not None
                else None
            )

        obj.matrix = check_load_param("matrix")
        obj.bias = check_load_param("bias")
        obj.idt = check_load_param("idt")
        return obj


MLP_ = make_multilayer_network(MLPLayer, nn.Module)


class MLP(MLP_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.ModuleList(self.layers)

    forward = MLP_.call


EmbeddingNet = make_embedding_network(MLP, MLPLayer)

FittingNet = make_fitting_network(EmbeddingNet, MLP, MLPLayer)


class NetworkCollection(DPNetworkCollection, nn.Module):
    """PyTorch implementation of NetworkCollection."""

    NETWORK_TYPE_MAP: ClassVar[Dict[str, type]] = {
        "network": MLP,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(self, *args, **kwargs):
        # init both two base classes
        DPNetworkCollection.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self.networks = self._networks = torch.nn.ModuleList(self._networks)
