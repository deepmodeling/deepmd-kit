# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)


class EnergyModel(DPModelCommon, make_model(DPAtomicModel)):
    pass
