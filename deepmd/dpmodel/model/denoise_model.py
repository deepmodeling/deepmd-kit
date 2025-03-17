# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model import (
    DPDenoiseAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPDenoiseModel_ = make_model(DPDenoiseAtomicModel)


@BaseModel.register("denoise")
class DenoiseModel(DPModelCommon, DPDenoiseModel_):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        DPModelCommon.__init__(self)
        DPDenoiseModel_.__init__(self, *args, **kwargs)