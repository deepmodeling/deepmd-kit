# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.pt.model.task.dipole import (
    DipoleFittingNet,
)
from deepmd.utils.preset_out_bias import (
    preset_assigns,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        if not isinstance(fitting, DipoleFittingNet):
            raise TypeError(
                "fitting must be an instance of DipoleFittingNet for DPDipoleAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
        if any(preset_assigns(self.preset_out_bias, key) for key in self.bias_keys):
            raise ValueError(
                "Dipole models do not apply an output bias and cannot use "
                "assigned preset_out_bias values."
            )

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # dipole not applying bias
        return ret
