# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
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

DPEnergyModel_ = make_model(DPAtomicModel)


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
