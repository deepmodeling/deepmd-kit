# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.model.polar_model import PolarModel as PolarModelDP
from deepmd.jax.atomic_model.polar_atomic_model import (
    DPAtomicModelPolar,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_model import (
    make_jax_dp_model_from_dpmodel,
)


@BaseModel.register("polar")
class PolarModel(make_jax_dp_model_from_dpmodel(PolarModelDP, DPAtomicModelPolar)):
    pass
