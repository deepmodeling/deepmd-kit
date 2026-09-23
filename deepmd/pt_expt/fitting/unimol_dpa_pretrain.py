# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.fitting.unimol_dpa_pretrain import (
    UniMolDPAPretrainFitting as UniMolDPAPretrainFittingDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)

# The coordinate head projects with the descriptor's own ``SO3Linear``, which
# keeps its weights as plain arrays. On this backend ``dpmodel_setattr``
# registers a plain array as a *buffer*, so without this the head would hold no
# parameters at all: the optimizer is built from ``wrapper.parameters()``, so it
# would train as a frozen random projection, and silently -- the coordinate term
# still falls, because gradients reach the backbone through it either way. The
# DPA4 descriptors each carry their own copy of this promotion for the same
# reason; this is the fitting's.
_TRAINABLE_ATTRS: dict[str, tuple[str, ...]] = {
    "SO3Linear": ("weight", "bias"),
}


def _promote_trainable_tree(module: torch.nn.Module) -> torch.nn.Module:
    """Re-register the float buffers that are meant to be learned.

    Runs after the tree is built: deserialization assigns arrays onto nested
    attributes, which would be registered as buffers again.
    """
    for sub in module.modules():
        if not getattr(sub, "trainable", True):
            continue
        for name in _TRAINABLE_ATTRS.get(type(sub).__name__, ()):
            buf = sub._buffers.get(name)
            if buf is None or not buf.is_floating_point():
                continue
            del sub._buffers[name]
            setattr(sub, name, torch.nn.Parameter(buf, requires_grad=True))
    return module


@BaseFitting.register("unimol_dpa_pretrain")
@torch_module
class UniMolDPAPretrainFitting(UniMolDPAPretrainFittingDP):
    """Uni-Mol's heads on a DPA backbone, PyTorch-Exportable backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _promote_trainable_tree(self)

    @classmethod
    def deserialize(cls, data: dict) -> "UniMolDPAPretrainFitting":
        """Deserialize the fitting, keeping its learned weights learnable."""
        return _promote_trainable_tree(super().deserialize(data))
