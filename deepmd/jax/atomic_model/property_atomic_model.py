# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.property_atomic_model import (
    DPPropertyAtomicModel as DPAtomicModelPropertyDP,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    make_jax_dp_atomic_model_from_dpmodel,
)


class DPAtomicModelProperty(
    make_jax_dp_atomic_model_from_dpmodel(DPAtomicModelPropertyDP)
):
    pass
