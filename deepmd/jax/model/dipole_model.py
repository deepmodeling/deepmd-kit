# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.model.dipole_model import DipoleModel as DipoleModelDP
from deepmd.jax.atomic_model.dipole_atomic_model import (
    DPAtomicModelDipole,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_model import (
    make_jax_dp_model_from_dpmodel,
)


@BaseModel.register("dipole")
class DipoleModel(make_jax_dp_model_from_dpmodel(DipoleModelDP, DPAtomicModelDipole)):
    pass
