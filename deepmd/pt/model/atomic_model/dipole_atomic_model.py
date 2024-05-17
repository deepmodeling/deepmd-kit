# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

from deepmd.pt.model.task.dipole import (
    DipoleFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

if TYPE_CHECKING:
    import torch


class DPDipoleAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, DipoleFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        # dipole not applying bias
        return ret
