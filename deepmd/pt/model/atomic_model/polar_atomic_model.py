# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
)

import torch

from deepmd.pt.model.task.polarizability import (
    PolarFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPolarAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, PolarFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: Dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        # TODO: migrate bias
        return ret
