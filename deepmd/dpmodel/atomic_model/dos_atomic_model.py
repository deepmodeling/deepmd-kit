# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.fitting.dos_fitting import (
    DOSFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDOSAtomicModel(DPAtomicModel):
    def __init__(
        self,
        descriptor: BaseDescriptor,
        fitting: BaseFitting,
        type_map: list[str],
        **kwargs: Any,
    ) -> None:
        if not isinstance(fitting, DOSFittingNet):
            raise TypeError(
                "fitting must be an instance of DOSFittingNet for DPDOSAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
