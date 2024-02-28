# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .make_model import (
    make_model,
)


# use "class" to resolve "Variable not allowed in type expression"
@BaseModel.register("standard")
class DPModel(make_model(DPAtomicModel), BaseModel):
    def data_requirement(self) -> dict:
        """Get the data requirement for the model."""
        raise NotImplementedError
