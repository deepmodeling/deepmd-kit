# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.fitting.dipole_fitting import (
    DipoleFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    def __init__(
        self, descriptor: Any, fitting: Any, type_map: list[str], **kwargs: Any
    ) -> None:
        if not isinstance(fitting, DipoleFitting):
            raise TypeError(
                "fitting must be an instance of DipoleFitting for DPDipoleAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, np.ndarray],
        atype: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # dipole not applying bias
        return ret
