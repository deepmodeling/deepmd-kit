# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task.dipole import (
    DipoleFittingNet,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
    EnergyFittingNetDirect,
)
from deepmd.pt.model.task.polarizability import (
    PolarFittingNet,
)

from .make_model import (
    make_model,
)


@BaseModel.register("standard")
class DPModel(make_model(DPAtomicModel), BaseModel):
    def __new__(cls, descriptor, fitting, *args, **kwargs):
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
            if isinstance(fitting, EnergyFittingNet) or isinstance(
                fitting, EnergyFittingNetDirect
            ):
                cls = EnergyModel
            elif isinstance(fitting, DipoleFittingNet):
                cls = DipoleModel
            elif isinstance(fitting, PolarFittingNet):
                cls = PolarModel
            # else: unknown fitting type, fall back to DPModel
        return super().__new__(cls)

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["descriptor"] = BaseDescriptor.update_sel(
            global_jdata, local_jdata["descriptor"]
        )
        return local_jdata_cpy
