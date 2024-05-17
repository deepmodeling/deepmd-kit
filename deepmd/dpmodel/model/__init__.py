# SPDX-License-Identifier: LGPL-3.0-or-later
"""The model that takes the coordinates, cell and atom types as input
and predicts some property. The models are automatically generated from
atomic models by the `deepmd.dpmodel.make_model` method.

The `make_model` method does the reduction, auto-differentiation
(dummy for dpmodels) and communication of the atomic properties
according to output variable definition
`deepmd.dpmodel.OutputVariableDef`.

All models should be inherited from :class:`deepmd.dpmodel.model.base_model.BaseModel`.
Models generated by `make_model` have already done it.
"""

from __future__ import (
    annotations,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)
from .spin_model import (
    SpinModel,
)

__all__ = [
    "DPModelCommon",
    "SpinModel",
    "make_model",
]
