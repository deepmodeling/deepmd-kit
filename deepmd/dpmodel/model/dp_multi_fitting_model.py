# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.dp_multi_fitting_atomic_model import (
    DPMultiFittingAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_multi_fitting_model import (
    make_multi_fitting_model,
)

DPMultiFittingModel_ = make_multi_fitting_model(DPMultiFittingAtomicModel)


@BaseModel.register("multi_fitting")
class DPMultiFittingModel(DPModelCommon, DPMultiFittingModel_):
    model_type = "multi_fitting"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPMultiFittingModel_.__init__(self, *args, **kwargs)
