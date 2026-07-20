# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model import EnergyModel as EnergyModelDP
from deepmd.jax.atomic_model.energy_atomic_model import (
    DPAtomicModelEnergy,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_model import (
    make_jax_dp_model_from_dpmodel,
)


@BaseModel.register("ener")
class EnergyModel(make_jax_dp_model_from_dpmodel(EnergyModelDP, DPAtomicModelEnergy)):
    pass
