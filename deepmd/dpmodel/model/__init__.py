# SPDX-License-Identifier: LGPL-3.0-or-later
"""The model that takes the coordinates, cell and atom types as input
and predicts some property. The models are automatically generated from
atomic models by the `deepmd.dpmodel.make_model` method.

The `make_model` method does the reduction, auto-differentiation
(dummy for dpmodels) and communication of the atomic properties
according to output variable definition
`deepmd.dpmodel.OutputVariableDef`.

"""

from .dp_model import (
    DPModel,
)
from .make_model import (
    make_model,
)
from .spin_model import (
    SpinModel,
)

__all__ = [
    "DPModel",
    "SpinModel",
    "make_model",
]
