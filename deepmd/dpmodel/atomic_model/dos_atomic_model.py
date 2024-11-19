# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.dos_fitting import (
    DOSFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDOSAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, DOSFittingNet):
            raise TypeError(
                "fitting must be an instance of DOSFittingNet for DPDOSAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
