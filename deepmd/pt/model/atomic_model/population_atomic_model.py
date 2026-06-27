# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.pt.model.task.population import (
    PopulationFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPopulationAtomicModel(DPAtomicModel):
    """Atomic model for population fitting, wrapping PopulationFittingNet."""

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        """Initialize DPPopulationAtomicModel, enforcing PopulationFittingNet."""
        if not isinstance(fitting, PopulationFittingNet):
            raise TypeError(
                "fitting must be an instance of PopulationFittingNet for DPPopulationAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Don't apply bias for population fitting."""
        return ret
