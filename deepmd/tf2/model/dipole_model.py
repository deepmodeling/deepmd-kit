# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.model.dipole_model import DipoleModel as DipoleModelDP
from deepmd.tf2.atomic_model.dipole_atomic_model import (
    DPAtomicModelDipole,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.model.dp_model import (
    make_tf2_dp_model_from_dpmodel,
)


@BaseModel.register("dipole")
class DipoleModel(make_tf2_dp_model_from_dpmodel(DipoleModelDP, DPAtomicModelDipole)):
    pass
