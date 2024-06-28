# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, PropertyFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_intensive(self) -> bool:
        """Get whether the property is intensive."""
        return self.atomic_output_def()["property"].get_intensive()