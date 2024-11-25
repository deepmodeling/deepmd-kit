# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.dpmodel.fitting.dipole_fitting import (
    DipoleFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, DipoleFitting):
            raise TypeError(
                "fitting must be an instance of DipoleFitting for DPDipoleAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, np.ndarray],
        atype: np.ndarray,
    ):
        # dipole not applying bias
        return ret
