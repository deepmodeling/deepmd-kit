# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.kernels.cuda.graph_fitting import (
    fitting_eligible,
    graph_fitting,
)
from deepmd.kernels.cuda.graph_fitting import op_available as cuda_fitting_available
from deepmd.kernels.utils import (
    cuda_infer_level,
)
from deepmd.pt_expt.common import (
    torch_module,
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
        """Graph-native fitting forward, fused on CUDA when eligible.

        At ``DP_CUDA_INFER >= 1`` an inference-mode call on an eligible
        network (see
        :func:`~deepmd.kernels.cuda.graph_fitting.fitting_eligible`)
        routes through the fused cuBLAS operator; anything else keeps the
        dpmodel reference. Routing is device-free, so a CPU ``make_fx`` trace
        bakes the operator into the exported graph.
        """
        if (
            not self.training
            and fparam is None
            and aparam is None
            and cuda_infer_level() >= 1
            and cuda_fitting_available()
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
