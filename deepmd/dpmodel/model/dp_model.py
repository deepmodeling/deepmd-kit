# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
)

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .make_model import (
    make_model,
)


# use "class" to resolve "Variable not allowed in type expression"
@BaseModel.register("standard")
class DPModel(make_model(DPAtomicModel), BaseModel):
    def data_requirement(self) -> List[DataRequirementItem]:
        """Get the data requirement for the model."""
        raise NotImplementedError

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
