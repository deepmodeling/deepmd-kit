# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.dos_atomic_model import (
    DPDOSAtomicModel as DPAtomicModelDOSDP,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    make_jax_dp_atomic_model_from_dpmodel,
)


class DPAtomicModelDOS(make_jax_dp_atomic_model_from_dpmodel(DPAtomicModelDOSDP)):
    pass
