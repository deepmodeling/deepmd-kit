# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
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
from deepmd.utils.version import (
    check_version_compatibility,
)

from .make_model import (
    make_model,
)


@BaseModel.register("standard")
class DPModel(make_model(DPAtomicModel)):
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

    def get_fitting_net(self):
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self):
        """Get the descriptor."""
        return self.atomic_model.descriptor

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # directly call the forward_common method when no specific transform rule
        return self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

    @classmethod
    def deserialize(cls, data) -> "DPAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class")
        data.pop("type")
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))
        fitting_obj = BaseFitting.deserialize(data.pop("fitting"))
        type_map = data.pop("type_map")
        obj = cls(descriptor_obj, fitting_obj, type_map=type_map, **data)
        return obj
