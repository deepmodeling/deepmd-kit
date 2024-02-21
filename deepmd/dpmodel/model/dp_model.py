# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)

from .make_model import (
    make_model,
)


# use "class" to resolve "Variable not allowed in type expression"
class DPModel(make_model(DPAtomicModel)):
    pass
