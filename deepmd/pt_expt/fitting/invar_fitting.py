# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel.fitting.invar_fitting import InvarFitting as InvarFittingDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)


@BaseFitting.register("invar")
@torch_module
class InvarFitting(InvarFittingDP):
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
