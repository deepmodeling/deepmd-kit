# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model import (
    DPDOSAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPDOSModel_ = make_model(DPDOSAtomicModel)


@BaseModel.register("dos")
class DOSModel(DPModelCommon, DPDOSModel_):
    model_type = "dos"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPDOSModel_.__init__(self, *args, **kwargs)
