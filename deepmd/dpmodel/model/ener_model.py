# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)

from deepmd.dpmodel.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel)


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
        self._enable_hessian = False
        self.hess_fitting_def = None

    def enable_hessian(self):
        self.hess_fitting_def = deepcopy(self.atomic_output_def())
        self.hess_fitting_def["energy"].r_hessian = True
        self._enable_hessian = True

    def atomic_output_def(self) -> FittingOutputDef:
        if self._enable_hessian:
            return self.hess_fitting_def
        return super().atomic_output_def()
