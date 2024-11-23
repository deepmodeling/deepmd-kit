# SPDX-License-Identifier: LGPL-3.0-or-later


from deepmd.dpmodel.atomic_model import (
    DPDipoleAtomicModel,
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

DPDipoleModel_ = make_model(DPDipoleAtomicModel)


@BaseModel.register("dipole")
class DipoleModel(DPModelCommon, DPDipoleModel_):
    model_type = "dipole"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPDipoleModel_.__init__(self, *args, **kwargs)
