# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
)

import torch

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    def __init__(self, *args, **kwargs):
        assert isinstance(fitting, DipoleFittingNet)
        super().__init__(*args, **kwargs)
        
    def apply_out_stat(
        self,
        ret: Dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        # dipole not applying bias
        return ret
