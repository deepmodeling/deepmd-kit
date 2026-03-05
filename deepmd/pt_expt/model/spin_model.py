# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.model.spin_model import SpinModel as SpinModelDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.utils.spin import (
    Spin,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)


@torch_module
class SpinModel(SpinModelDP):
    def __getattr__(self, name: str) -> Any:
        """Get attribute from the wrapped model.

        In torch.nn.Module, submodules are stored in _modules, not __dict__.
        Override the dpmodel version to use torch.nn.Module's __getattr__
        first (which checks _parameters, _buffers, _modules), then fall
        back to backbone_model delegation for arbitrary attributes.
        """
        try:
            return torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        # backbone_model is in _modules, access via _modules directly
        # to avoid re-entering __getattr__
        modules = self.__dict__.get("_modules", {})
        backbone = modules.get("backbone_model")
        if backbone is not None:
            return getattr(backbone, name)
        raise AttributeError(name)

    @classmethod
    def deserialize(cls, data: dict) -> "SpinModel":
        from deepmd.dpmodel.atomic_model import (
            DPEnergyAtomicModel,
        )

        backbone_model_obj = make_model(
            DPEnergyAtomicModel, T_Bases=(BaseModel,)
        ).deserialize(data["backbone_model"])
        spin = Spin.deserialize(data["spin"])
        return cls(
            backbone_model=backbone_model_obj,
            spin=spin,
        )
