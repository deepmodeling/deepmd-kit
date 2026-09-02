# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.kernels.graph_fitting import (
    fitting_eligible,
    graph_fitting,
)
from deepmd.pt_expt.kernels.graph_fitting import op_available as fused_fitting_available
from deepmd.pt_expt.kernels.utils import (
    fused_operators_enabled,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("ener")
@torch_module
class EnergyFittingNet(EnergyFittingNetDP):
    def share_params(self, *args: Any, **kwargs: Any) -> None:
        from deepmd.pt_expt.fitting.invar_fitting import (
            InvarFitting,
        )

        return InvarFitting.share_params(self, *args, **kwargs)

    def call_graph(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Graph-native fitting forward, fused when the backend supports it.

        An inference-mode call on an eligible network (see
        :func:`~deepmd.pt_expt.kernels.graph_fitting.fitting_eligible`)
        routes through the fused operator of the backend device; anything else
        keeps the dpmodel reference. Routing resolves against the backend
        device rather than a traced tensor, because every export traces on CPU
        and moves the program afterwards.
        """
        if (
            not self.training
            and fparam is None
            and aparam is None
            and fused_operators_enabled()
            and fused_fitting_available()
            and fitting_eligible(self)
        ):
            return graph_fitting(self, descriptor, atype)
        return EnergyFittingNetDP.call_graph(
            self,
            descriptor,
            atype,
            gr=gr,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
