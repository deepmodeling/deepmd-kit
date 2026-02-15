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
    """Parameter subclass that supports ``np.array(param)`` conversion.

    Note: this class is intentionally NOT used for model parameters.
    ``make_fx`` (``torch.fx.experimental.proxy_tensor``) uses
    ``ProxyTorchDispatchMode`` to intercept tensor operations.  When an
    operand is a *subclass* of ``torch.Tensor`` (including subclasses of
    ``torch.nn.Parameter``), PyTorch invokes the ``__torch_function__``
    protocol which the proxy dispatch mode does not handle, causing
    ``aten.mm`` and other ops to fail with "Multiple dispatch failed …
    returned NotImplemented".  Using plain ``torch.nn.Parameter`` avoids
    this because the proxy mode is designed to work with the base
    ``Parameter`` type.  ``TorchArrayParam`` is kept only for backward
    compatibility and should not be used for new code.
    """

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
    """PyTorch layer wrapping dpmodel's ``NativeLayer``.

    Two aspects of the inherited dpmodel ``call()`` are incompatible with
    ``make_fx`` tracing (used to export ``forward_lower`` with
    ``autograd.grad``-based force/virial computation):

    1. **Ellipsis indexing** (``self.w[...]``):  On a ``torch.Tensor``
       this triggers ``aten.alias``, an op that ``ProxyTorchDispatchMode``
       does not support, resulting in "Multiple dispatch failed for
       ``aten.alias.default``".
    2. **``array_api_compat`` wrappers** (``xp = array_api_compat
       .array_namespace(x); xp.matmul(…)``):  The wrappers re-enter
       ``torch.matmul`` through Python, which goes through the
       ``__torch_function__`` protocol.  Under the proxy dispatch mode
       this path also fails with "Multiple dispatch failed".

    This class therefore overrides ``call()`` with an implementation that
    uses plain ``torch`` ops exclusively (``torch.matmul``, ``torch.tanh``,
    etc.), avoiding both issues.

    Trainable weights are stored as plain ``torch.nn.Parameter`` (not
    ``TorchArrayParam``) for the same ``make_fx`` compatibility reason —
    see the ``TorchArrayParam`` docstring.
    """

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
                    if isinstance(value, torch.nn.Parameter)
                    else torch.nn.Parameter(val, requires_grad=True)
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

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using pure torch ops.

        Overrides dpmodel's ``call()`` to ensure compatibility with
        ``make_fx`` (``torch.fx.experimental.proxy_tensor``).

        The dpmodel implementation uses ``self.w[...]`` and
        ``array_api_compat.array_namespace(x).matmul(…)`` for
        backend-agnostic array operations.  Both patterns break under
        ``make_fx``'s ``ProxyTorchDispatchMode``:

        - ``self.w[...]`` emits ``aten.alias`` which the proxy mode
          cannot dispatch.
        - ``array_api_compat`` re-enters ``torch.matmul`` via Python,
          hitting ``__torch_function__`` which the proxy mode returns
          ``NotImplemented`` for.

        This override uses ``torch.matmul``, ``torch.cat``, and
        ``_torch_activation`` directly, sidestepping both issues.
        """
        if self.w is None or self.activation_function is None:
            raise ValueError("w, b, and activation_function must be set")
        y = (
            torch.matmul(x, self.w) + self.b
            if self.b is not None
            else torch.matmul(x, self.w)
        )
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        y = _torch_activation(y, self.activation_function)
        if self.idt is not None:
            y = y * self.idt
        if self.resnet and self.w.shape[1] == self.w.shape[0]:
            y = y + x
        elif self.resnet and self.w.shape[1] == 2 * self.w.shape[0]:
            y = y + torch.cat([x, x], dim=-1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


def _torch_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply activation function using native torch ops.

    The dpmodel ``get_activation_fn`` returns closures that call
    ``array_api_compat.array_namespace(x).tanh(x)`` etc.  Under
    ``make_fx`` proxy tracing, the ``array_api_compat`` indirection
    triggers ``__torch_function__`` dispatch failures.  This function
    calls ``torch.tanh`` and friends directly to avoid the issue.
    """
    name = name.lower()
    if name == "tanh":
        return torch.tanh(x)
    elif name == "relu":
        return torch.relu(x)
    elif name in ("gelu", "gelu_tf"):
        return torch.nn.functional.gelu(x, approximate="tanh")
    elif name == "relu6":
        return torch.clamp(x, min=0.0, max=6.0)
    elif name == "softplus":
        return torch.nn.functional.softplus(x)
    elif name == "sigmoid":
        return torch.sigmoid(x)
    elif name == "silu":
        return torch.nn.functional.silu(x)
    elif name in ("none", "linear"):
        return x
    else:
        raise NotImplementedError(name)


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
