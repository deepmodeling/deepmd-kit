# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model.dos_model import DOSModel as DOSModelDP
from deepmd.jax.atomic_model.dos_atomic_model import (
    DPAtomicModelDOS,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_model import (
    make_jax_dp_model_from_dpmodel,
)


@BaseModel.register("dos")
class DOSModel(make_jax_dp_model_from_dpmodel(DOSModelDP, DPAtomicModelDOS)):
    pass
