# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt wrapper for the shared radial embedding MLP.

The dpmodel :class:`RadialMLP` stores its layers in a single ``net`` list whose
entries alternate between ``NativeOP`` modules (``NativeLayer`` linear maps and
``RMSNorm``) and a *plain activation function* returned by ``get_activation_fn``.
The generic ``dpmodel_setattr`` list conversion only turns a list into a
``torch.nn.ModuleList`` when every entry is a module (or ``None``); the bare
activation function makes that check fail, so the list -- and therefore the
linear / norm weights nested inside it -- would stay raw numpy arrays, invisible
to autograd and the optimizer.

This wrapper rebuilds ``net`` as a ``ModuleList`` once the tree is constructed,
converting each ``NativeOP`` entry through the standard registry (so the linear
maps become trainable :class:`~deepmd.pt_expt.utils.network.NativeLayer` weights
and the norms become promotable buffers) and replacing the plain activation with
a parameter-free torch module.  The activation reuses the backend's
``_torch_activation`` so it is bit-identical to the reference pt
``ActivationFn`` and safe under ``make_fx`` tracing.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialMLP as RadialMLPDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
    try_convert_module,
)
from deepmd.pt_expt.utils.network import (
    _torch_activation,
)


class _ScalarActivation(torch.nn.Module):
    """Parameter-free torch module applying a named scalar activation.

    Mirrors the position the plain activation function occupies in the dpmodel
    ``RadialMLP.net`` list, so the whole list can become a ``ModuleList``.
    """

    def __init__(self, activation_function: str) -> None:
        super().__init__()
        self.activation_function = str(activation_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _torch_activation(x, self.activation_function)


@torch_module
class RadialMLP(RadialMLPDP):
    """Radial embedding MLP with a torch-native, trainable ``net``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ``self.net`` is still the raw dpmodel list here (the bare activation
        # function blocked the generic list -> ModuleList conversion).  Convert
        # every entry explicitly so the linear / norm weights live in trainable
        # torch sub-modules.
        self.net = torch.nn.ModuleList(self._convert_layer(layer) for layer in self.net)

    def _convert_layer(self, layer: Any) -> torch.nn.Module:
        if isinstance(layer, torch.nn.Module):
            return layer
        if isinstance(layer, NativeOP):
            return try_convert_module(layer)
        return _ScalarActivation(self.activation_function)


# Build the torch-native RadialMLP wherever the dpmodel one is assigned in the
# auto-wrapped descriptor tree (e.g. ``DescrptDPA4.radial_embedding``).
register_dpmodel_mapping(RadialMLPDP, lambda v: RadialMLP.deserialize(v.serialize()))
