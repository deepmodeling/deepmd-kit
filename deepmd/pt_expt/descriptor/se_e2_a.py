# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeAArrayAPI as DescrptSeADP
from deepmd.pt_expt.common import (
    dpmodel_setattr,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e2_a_expt")
@BaseDescriptor.register("se_a_expt")
class DescrptSeA(DescrptSeADP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        DescrptSeADP.__init__(self, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Ensure torch.nn.Module.__call__ drives forward() for export/tracing.
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        handled, value = dpmodel_setattr(self, name, value)
        if not handled:
            super().__setattr__(name, value)

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        descrpt, rot_mat, g2, h2, sw = self.call(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        return descrpt, rot_mat, g2, h2, sw
