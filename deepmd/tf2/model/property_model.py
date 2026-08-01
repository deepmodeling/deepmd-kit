# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.model.property_model import PropertyModel as PropertyModelDP
from deepmd.tf2.atomic_model.property_atomic_model import (
    DPAtomicModelProperty,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.model.dp_model import (
    make_tf2_dp_model_from_dpmodel,
)


@BaseModel.register("property")
class PropertyModel(
    make_tf2_dp_model_from_dpmodel(PropertyModelDP, DPAtomicModelProperty)
):
    pass
