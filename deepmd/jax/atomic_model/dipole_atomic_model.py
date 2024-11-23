# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.dipole_atomic_model import (
    DPDipoleAtomicModel as DPAtomicModelDipoleDP,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    make_jax_dp_atomic_model_from_dpmodel,
)


class DPAtomicModelDipole(make_jax_dp_atomic_model_from_dpmodel(DPAtomicModelDipoleDP)):
    pass
