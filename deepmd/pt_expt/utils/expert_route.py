# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt router of the expert coefficients."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.expert_route import ConditionRouter as ConditionRouterDP
from deepmd.pt_expt.common import (
    promote_buffers,
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class ConditionRouter(ConditionRouterDP):
    """Router whose weights are parameters, trainable with the router."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._promote_parameters()

    def _promote_parameters(self) -> None:
        """Register the weights of the router as parameters."""
        promote_buffers(self, self.PARAMETER_ATTRIBUTES, self.trainable)

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "ConditionRouter":
        obj = super().deserialize(data)
        obj._promote_parameters()
        return obj

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.call(*args, **kwargs)


register_dpmodel_mapping(
    ConditionRouterDP,
    lambda v: ConditionRouter.deserialize(v.serialize()),
)
