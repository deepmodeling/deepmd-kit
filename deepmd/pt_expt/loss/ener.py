# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.loss.ener import EnergyLoss as EnergyLossDP
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class EnergyLoss(EnergyLossDP):
    pass
