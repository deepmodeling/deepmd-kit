# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model import (
    DPPropertyAtomicModel,
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

DPPropertyModel_ = make_model(DPPropertyAtomicModel)


@BaseModel.register("property")
class PropertyModel(DPModelCommon, DPPropertyModel_):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        DPModelCommon.__init__(self)
        DPPropertyModel_.__init__(self, *args, **kwargs)
