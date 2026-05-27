# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP
from deepmd.pt_expt.common import (
    dpmodel_setattr,
    register_dpmodel_mapping,
)

# Import network to ensure EmbeddingNet is registered before TypeEmbedNet is used
from deepmd.pt_expt.utils import network  # noqa: F401


class TypeEmbedNet(TypeEmbedNetDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        TypeEmbedNetDP.__init__(self, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Ensure torch.nn.Module.__call__ drives forward() for export/tracing.
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        # Use common dpmodel_setattr which handles embedding_net conversion via registry
        handled, value = dpmodel_setattr(self, name, value)
        if not handled:
            super().__setattr__(name, value)

    def forward(self) -> torch.Tensor:
        # Call dpmodel's implementation (now with proper device handling)
        return self.call()


register_dpmodel_mapping(
    TypeEmbedNetDP,
    lambda v: TypeEmbedNet.deserialize(v.serialize()),
)
