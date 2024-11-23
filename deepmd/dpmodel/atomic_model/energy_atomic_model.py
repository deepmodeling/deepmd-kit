# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.ener_fitting import (
    EnergyFittingNet,
    InvarFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPEnergyAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not (
            isinstance(fitting, EnergyFittingNet) or isinstance(fitting, InvarFitting)
        ):
            raise TypeError(
                "fitting must be an instance of EnergyFittingNet or InvarFitting for DPEnergyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
