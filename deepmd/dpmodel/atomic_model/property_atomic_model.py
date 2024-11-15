# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs) -> None:
        assert isinstance(fitting, PropertyFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)
