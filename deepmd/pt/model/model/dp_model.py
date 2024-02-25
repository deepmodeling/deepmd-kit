# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task.dipole import (
    DipoleFittingNet,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
)
from deepmd.pt.model.task.polarizability import (
    PolarFittingNet,
)

from .make_model import (
    make_model,
)


@BaseModel.register("standard")
class DPModel(make_model(DPAtomicModel), BaseModel):
    def __new__(cls, descriptor, fitting, **kwargs):
        from deepmd.pt.model.model.dipole_model import (
            DipoleModel,
        )
        from deepmd.pt.model.model.ener_model import (
            EnergyModel,
        )
        from deepmd.pt.model.model.polar_model import (
            PolarModel,
        )

        # according to the fitting network to decide the type of the model
        if cls is DPModel:
            # map fitting to model
            if isinstance(fitting, EnergyFittingNet):
                cls = EnergyModel
            elif isinstance(fitting, DipoleFittingNet):
                cls = DipoleModel
            elif isinstance(fitting, PolarFittingNet):
                cls = PolarModel
            else:
                raise ValueError(f"Unknown fitting type {fitting}")
        return super().__new__(cls)
