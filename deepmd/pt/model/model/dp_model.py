# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
)

from .make_model import (
    make_model,
)

DPModel = make_model(DPAtomicModel)
