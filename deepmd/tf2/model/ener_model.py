# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model import EnergyModel as EnergyModelDP
from deepmd.tf2.atomic_model.energy_atomic_model import (
    DPAtomicModelEnergy,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.model.dp_model import (
    make_tf2_dp_model_from_dpmodel,
)


@BaseModel.register("sezm_ener")
@BaseModel.register("dpa4_ener")
@BaseModel.register("ener")
class EnergyModel(make_tf2_dp_model_from_dpmodel(EnergyModelDP, DPAtomicModelEnergy)):
    pass
