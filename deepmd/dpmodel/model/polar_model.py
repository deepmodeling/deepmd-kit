# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.atomic_model import (
    DPPolarAtomicModel,
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

DPPolarModel_ = make_model(DPPolarAtomicModel)


@BaseModel.register("polar")
class PolarModel(DPModelCommon, DPPolarModel_):
    model_type = "polar"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPPolarModel_.__init__(self, *args, **kwargs)
