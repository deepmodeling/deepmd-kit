# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

import numpy as np
import torch

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.utils.network import LayerNorm as LayerNormDP
from deepmd.dpmodel.utils.network import NativeLayer as NativeLayerDP
from deepmd.dpmodel.utils.network import NetworkCollection as NetworkCollectionDP
from deepmd.dpmodel.utils.network import (
    make_embedding_network,
    make_fitting_network,
    make_multilayer_network,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    to_torch_array,
)


class TorchArrayParam(torch.nn.Parameter):
    def __new__(  # noqa: PYI034
        cls, data: Any = None, requires_grad: bool = True
    ) -> "TorchArrayParam":
        return torch.nn.Parameter.__new__(cls, data, requires_grad)

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        arr = self.detach().cpu().numpy()
        if dtype is None:
            return arr
        return arr.astype(dtype)


class NativeLayer(NativeLayerDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        NativeLayerDP.__init__(self, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"w", "b", "idt"} and "_parameters" in self.__dict__:
            val = to_torch_array(value)
            if val is None:
                if name in self._parameters:
                    self._parameters[name] = None
                    return
                if name in self._buffers:
                    self._buffers[name] = None
                    return
                return super().__setattr__(name, None)
            if getattr(self, "trainable", False):
                param = (
                    value
                    if isinstance(value, TorchArrayParam)
                    else TorchArrayParam(val, requires_grad=True)
                )
                if name in self._parameters:
                    self._parameters[name] = param
                    return
                return super().__setattr__(name, param)
            if name in self._buffers:
                self._buffers[name] = val
                return
            # Register on first assignment so tensors are in state_dict and moved by .to().
            self.register_buffer(name, val)
            return
        return super().__setattr__(name, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


class NativeNet(make_multilayer_network(NativeLayer, NativeOP), torch.nn.Module):
    def __init__(self, layers: list[dict] | None = None) -> None:
        torch.nn.Module.__init__(self)
        super().__init__(layers)
        self.layers = torch.nn.ModuleList(self.layers)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


class EmbeddingNet(make_embedding_network(NativeNet, NativeLayer)):
    pass


class FittingNet(make_fitting_network(EmbeddingNet, NativeNet, NativeLayer)):
    pass


class NetworkCollection(NetworkCollectionDP, torch.nn.Module):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        self._module_networks = torch.nn.ModuleDict()
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: int | tuple, value: Any) -> None:
        idx = self._convert_key(key)
        super().__setitem__(key, value)
        net = self._networks[idx]
        key_str = str(idx)
        if isinstance(net, torch.nn.Module):
            self._module_networks[key_str] = net
        elif key_str in self._module_networks:
            del self._module_networks[key_str]


register_dpmodel_mapping(
    NetworkCollectionDP,
    lambda v: NetworkCollection.deserialize(v.serialize()),
)


class LayerNorm(LayerNormDP, NativeLayer):
    pass
