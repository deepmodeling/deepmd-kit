# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
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
