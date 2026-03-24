# SPDX-License-Identifier: LGPL-3.0-or-later
"""The atomic model provides the prediction of some property on each
atom.  All the atomic models are not supposed to be directly accessed
by users, but it provides a convenient interface for the
implementation of models.

Taking the energy models for example, the developeres only needs to
implement the atomic energy prediction via an atomic model, and the
model can be automatically made by the `deepmd.dpmodel.make_model`
method. The `DPModel` is made by
```
DPModel = make_model(DPAtomicModel)
```

"""

from .base_atomic_model import (
    BaseAtomicModel,
)
from .dipole_atomic_model import (
    DPDipoleAtomicModel,
)
from .dos_atomic_model import (
    DPDOSAtomicModel,
)
from .dp_atomic_model import (
    DPAtomicModel,
)
from .energy_atomic_model import (
    DPEnergyAtomicModel,
)
from .linear_atomic_model import (
    DPZBLLinearEnergyAtomicModel,
    LinearEnergyAtomicModel,
)
from .pairtab_atomic_model import (
    PairTabAtomicModel,
)
from .polar_atomic_model import (
    DPPolarAtomicModel,
)
from .property_atomic_model import (
    DPPropertyAtomicModel,
)

__all__ = [
    "BaseAtomicModel",
    "DPAtomicModel",
    "DPDOSAtomicModel",
    "DPDipoleAtomicModel",
    "DPEnergyAtomicModel",
    "DPPolarAtomicModel",
    "DPPropertyAtomicModel",
    "DPZBLLinearEnergyAtomicModel",
    "LinearEnergyAtomicModel",
    "PairTabAtomicModel",
]
