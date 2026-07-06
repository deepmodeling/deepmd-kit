# SPDX-License-Identifier: LGPL-3.0-or-later
from .denoise import (
    DenoiseLoss,
)
from .dens import (
    DeNSLoss,
)
from .dos import (
    DOSLoss,
)
from .ener import (
    EnergyHessianStdLoss,
    EnergyStdLoss,
)
from .ener_spin import (
    EnergySpinLoss,
)
from .group_property import (
    GroupPropertyLoss,
)
from .loss import (
    TaskLoss,
)
from .population import (
    PopulationLoss,
)
from .property import (
    PropertyLoss,
)
from .tensor import (
    TensorLoss,
)

__all__ = [
    "DOSLoss",
    "DeNSLoss",
    "DenoiseLoss",
    "EnergyHessianStdLoss",
    "EnergySpinLoss",
    "EnergyStdLoss",
    "GroupPropertyLoss",
    "PopulationLoss",
    "PropertyLoss",
    "TaskLoss",
    "TensorLoss",
]
