# SPDX-License-Identifier: LGPL-3.0-or-later
from .denoise import (
    DenoiseLoss,
)
from .dos import (
    DOSLoss,
)
from .xas import (
    XASLoss,
)
from .ener import (
    EnergyHessianStdLoss,
    EnergyStdLoss,
)
from .ener_spin import (
    EnergySpinLoss,
)
from .loss import (
    TaskLoss,
)
from .property import (
    PropertyLoss,
)
from .tensor import (
    TensorLoss,
)

__all__ = [
    "DOSLoss",
    "XASLoss",
    "DenoiseLoss",
    "EnergyHessianStdLoss",
    "EnergySpinLoss",
    "EnergyStdLoss",
    "PropertyLoss",
    "TaskLoss",
    "TensorLoss",
]
