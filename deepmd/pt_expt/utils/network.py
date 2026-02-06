# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
from typing import (
    Any,
    ClassVar,
    Self,
)

import numpy as np

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
    to_torch_array,
)

torch = importlib.import_module("torch")


class TorchArrayParam(torch.nn.Parameter):
    def __new__(cls, data: Any = None, requires_grad: bool = True) -> Self:
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
        for name in ("w", "b", "idt"):
            if name in self._parameters or name in self._buffers:
                continue
            val = to_torch_array(getattr(self, name))
            if val is None:
                continue
            if self.trainable:
                if hasattr(self, name) and name not in self._parameters:
                    delattr(self, name)
                self.register_parameter(name, TorchArrayParam(val, requires_grad=True))
            else:
                if hasattr(self, name) and name not in self._buffers:
                    delattr(self, name)
                self.register_buffer(name, val)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"w", "b", "idt"} and "_parameters" in self.__dict__:
            val = to_torch_array(value)
            if val is None:
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
            return super().__setattr__(name, val)
        return super().__setattr__(name, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


class NativeNet(make_multilayer_network(NativeLayer, NativeOP), torch.nn.Module):
    def __init__(self, layers: list[dict] | None = None) -> None:
        torch.nn.Module.__init__(self)
        super().__init__(layers)
        self.layers = torch.nn.ModuleList(self.layers)

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
        super().__init__(*args, **kwargs)
        self._module_networks = torch.nn.ModuleDict()
        for idx, net in enumerate(self._networks):
            if isinstance(net, torch.nn.Module):
                self._module_networks[str(idx)] = net

    def __setitem__(self, key: int | tuple, value: Any) -> None:
        super().__setitem__(key, value)
        if isinstance(value, torch.nn.Module):
            self._module_networks[str(self._convert_key(key))] = value


class LayerNorm(LayerNormDP, NativeLayer):
    pass
