# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.common import (
    NativeOP,
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
        # Skip FrozenModelDP.__init__ which would load via Backend detection;
        # pt_expt handles .pte natively and re-deserializes other formats itself.
        NativeOP.__init__(self)
        self.model_file = model_file
        if model_file.endswith(".pte"):
            from deepmd.pt_expt.utils.serialization import (
                serialize_from_file,
            )

            data = serialize_from_file(model_file)
            self.model = BaseModel.deserialize(data["model"])
        else:
            from deepmd.backend.backend import (
                Backend,
            )

            inp_backend: Backend = Backend.detect_backend_by_model(model_file)()
            data = inp_backend.serialize_hook(model_file)
            self.model = BaseModel.deserialize(data["model"])
        self.model.eval()
