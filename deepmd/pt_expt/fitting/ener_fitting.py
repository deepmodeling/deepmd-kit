# SPDX-License-Identifier: LGPL-3.0-or-later

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
    pass
