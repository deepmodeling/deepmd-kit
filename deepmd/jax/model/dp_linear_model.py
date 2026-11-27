# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable linear compositions of JAX atomic energy models."""

from deepmd.dpmodel.model.dp_linear_model import (
    LinearEnergyModel as LinearEnergyModelDP,
)
from deepmd.jax.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_model import (
    make_jax_dp_model_from_dpmodel,
)
from deepmd.jax.model.ener_model import (
    EnergyModel,
)


@BaseModel.register("linear_ener")
@BaseModel.register("linear")
class LinearEnergyModel(
    make_jax_dp_model_from_dpmodel(LinearEnergyModelDP, LinearEnergyAtomicModel)
):
    _enable_hessian = False
    call = EnergyModel.call
    call_lower = EnergyModel.call_lower
    translated_output_def = EnergyModel.translated_output_def
