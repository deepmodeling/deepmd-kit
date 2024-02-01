# SPDX-License-Identifier: LGPL-3.0-or-later
from .dp_atomic_model import (
    DPAtomicModel,
)
from .make_model import (
    make_model,
)

DPModel = make_model(DPAtomicModel)
