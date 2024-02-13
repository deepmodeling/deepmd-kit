# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel import (
    DPAtomicModel,
)

from .make_model import (
    make_model,
)

DPModel = make_model(DPAtomicModel)
