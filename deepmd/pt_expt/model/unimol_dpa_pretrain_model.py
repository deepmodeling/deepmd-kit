# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

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

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the heads on a frame; a periodic cell is refused."""
        if box is not None and bool(torch.any(box != 0)):
            raise ValueError(
                "the unimol objective does not support periodic boundaries: its "
                "distance target is built from plain coordinate differences with "
                "no minimum-image convention, and its coverage keeps only local "
                "neighbours, so a cell would train against wrong labels rather "
                "than fail. Pass box=None"
            )
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        return {k: v for k, v in model_ret.items() if v is not None}
