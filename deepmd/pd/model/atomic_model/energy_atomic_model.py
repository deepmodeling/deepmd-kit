# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.pd.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pd.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pd.model.task.ener import (
    EnergyFittingNet,
    InvarFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPEnergyAtomicModel(DPAtomicModel):
    def __init__(
        self,
        descriptor: BaseDescriptor,
        fitting: BaseFitting,
        type_map: list[str],
        **kwargs: Any,
    ) -> None:
        assert isinstance(fitting, EnergyFittingNet) or isinstance(
            fitting, InvarFitting
        )
        super().__init__(descriptor, fitting, type_map, **kwargs)
