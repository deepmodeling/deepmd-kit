# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pd.model.task.ener import (
    EnergyFittingNet,
    InvarFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPEnergyAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, EnergyFittingNet) or isinstance(
            fitting, InvarFitting
        )
        super().__init__(descriptor, fitting, type_map, **kwargs)
