# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.model.polar_model import PolarModel as PolarModelDP
from deepmd.tf2.atomic_model.polar_atomic_model import (
    DPAtomicModelPolar,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.model.dp_model import (
    make_tf2_dp_model_from_dpmodel,
)


@BaseModel.register("polar")
class PolarModel(make_tf2_dp_model_from_dpmodel(PolarModelDP, DPAtomicModelPolar)):
    pass
