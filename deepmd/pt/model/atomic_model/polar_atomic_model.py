# SPDX-License-Identifier: LGPL-3.0-or-later
from .dp_atomic_model import (
    DPAtomicModel,
)
from typing import Dict
import torch

class DPPolarAtomicModel(DPAtomicModel):
    def apply_out_stat(
        self,
        ret: Dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        # TODO: migrate bias
        return ret
