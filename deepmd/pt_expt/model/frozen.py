# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.model.frozen import FrozenModel as FrozenModelDP
from deepmd.pt_expt.common import (
    torch_module,
)

from .model import (
    BaseModel,
)


@BaseModel.register("frozen")
@torch_module
class FrozenModel(FrozenModelDP):
    def __init__(self, model_file: str, **kwargs: Any) -> None:
        super().__init__(model_file, **kwargs)
        # Re-deserialize as a pt_expt model (parent creates a dpmodel model)
        self.model = BaseModel.deserialize(self.model.serialize())
        self.model.eval()
