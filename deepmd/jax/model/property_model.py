# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.model.property_model import PropertyModel as PropertyModelDP
from deepmd.jax.atomic_model.property_atomic_model import (
    DPAtomicModelProperty,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_model import (
    make_jax_dp_model_from_dpmodel,
)


@BaseModel.register("property")
class PropertyModel(
    make_jax_dp_model_from_dpmodel(PropertyModelDP, DPAtomicModelProperty)
):
    pass
