# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.model import EnergyModel as EnergyModelDP
from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)


@BaseModel.register("ener")
@flax_module
class EnergyModel(EnergyModelDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "atomic_model":
            value = DPAtomicModel.deserialize(value.serialize())
        return super().__setattr__(name, value)
