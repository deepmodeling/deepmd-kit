# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.polar_atomic_model import (
    DPPolarAtomicModel as DPAtomicModelPolarDP,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    make_jax_dp_atomic_model_from_dpmodel,
)


class DPAtomicModelPolar(make_jax_dp_atomic_model_from_dpmodel(DPAtomicModelPolarDP)):
    pass
