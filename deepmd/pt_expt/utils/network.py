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
from deepmd.dpmodel.utils.network import EmbeddingNet as EmbeddingNetDP
from deepmd.dpmodel.utils.network import FittingNet as FittingNetDP
from deepmd.dpmodel.utils.network import LayerNorm as LayerNormDP
from deepmd.dpmodel.utils.network import NativeLayer as NativeLayerDP
from deepmd.dpmodel.utils.network import NetworkCollection as NetworkCollectionDP
from deepmd.dpmodel.utils.network import (
    make_multilayer_network,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    to_torch_array,
    torch_module,
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


# do not apply torch_module until its setattr working to register parameters
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


@torch_module
class NativeNet(make_multilayer_network(NativeLayer, NativeOP)):
    def __init__(self, layers: list[dict] | None = None) -> None:
        super().__init__(layers)
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


class EmbeddingNet(EmbeddingNetDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        EmbeddingNetDP.__init__(self, *args, **kwargs)
        # EmbeddingNetDP.__init__ creates dpmodel NativeLayer instances.
        # Convert to pt_expt NativeLayer and wrap in ModuleList.
        self.layers = torch.nn.ModuleList(
            [NativeLayer.deserialize(layer.serialize()) for layer in self.layers]
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


register_dpmodel_mapping(
    EmbeddingNetDP,
    lambda v: EmbeddingNet.deserialize(v.serialize()),
)


class FittingNet(FittingNetDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        FittingNetDP.__init__(self, *args, **kwargs)
        # Convert dpmodel layers to pt_expt NativeLayer
        self.layers = torch.nn.ModuleList(
            [NativeLayer.deserialize(layer.serialize()) for layer in self.layers]
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


register_dpmodel_mapping(
    FittingNetDP,
    lambda v: FittingNet.deserialize(v.serialize()),
)


@torch_module
class NetworkCollection(NetworkCollectionDP):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
