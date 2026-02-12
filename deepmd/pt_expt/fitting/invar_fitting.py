# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.fitting.invar_fitting import InvarFitting as InvarFittingDP
from deepmd.pt_expt.common import (
    dpmodel_setattr,
    register_dpmodel_mapping,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.pt_expt.utils.network import (
    NetworkCollection,
)


@BaseFitting.register("invar")
class InvarFitting(InvarFittingDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        InvarFittingDP.__init__(self, *args, **kwargs)
        # Convert dpmodel NetworkCollection to pt_expt NetworkCollection
        self.nets = NetworkCollection.deserialize(self.nets.serialize())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Ensure torch.nn.Module.__call__ drives forward() for export/tracing.
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        handled, value = dpmodel_setattr(self, name, value)
        if not handled:
            super().__setattr__(name, value)

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.call(
            descriptor,
            atype,
            gr=gr,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )


register_dpmodel_mapping(
    InvarFittingDP,
    lambda v: InvarFitting.deserialize(v.serialize()),
)
