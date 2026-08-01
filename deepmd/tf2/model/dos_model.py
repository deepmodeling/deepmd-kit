# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model.dos_model import DOSModel as DOSModelDP
from deepmd.tf2.atomic_model.dos_atomic_model import (
    DPAtomicModelDOS,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.model.dp_model import (
    make_tf2_dp_model_from_dpmodel,
)


@BaseModel.register("dos")
class DOSModel(make_tf2_dp_model_from_dpmodel(DOSModelDP, DPAtomicModelDOS)):
    pass
