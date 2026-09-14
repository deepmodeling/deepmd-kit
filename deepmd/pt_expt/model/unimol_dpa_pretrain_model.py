# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model import (
    DPUniMolDPAAtomicModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)

DPUniMolDPAPretrainModel_ = make_model(DPUniMolDPAAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("unimol_dpa_pretrain")
class UniMolDPAPretrainModel(DPModelCommon, DPUniMolDPAPretrainModel_):
    """Uni-Mol's objective on a DPA backbone, PyTorch-Exportable backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        DPUniMolDPAPretrainModel_.__init__(self, *args, **kwargs)
