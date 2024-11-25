# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.energy_atomic_model import (
    DPEnergyAtomicModel as DPAtomicModelEnergyDP,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    make_jax_dp_atomic_model_from_dpmodel,
)


class DPAtomicModelEnergy(make_jax_dp_atomic_model_from_dpmodel(DPAtomicModelEnergyDP)):
    pass
