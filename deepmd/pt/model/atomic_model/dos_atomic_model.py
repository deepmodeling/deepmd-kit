# SPDX-License-Identifier: LGPL-3.0-or-later
from .dp_atomic_model import (
    DPAtomicModel,
)

from deepmd.pt.model.task.dos import (
    DOSFittingNet,
)

class DPDOSAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, DOSFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)
