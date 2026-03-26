# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.loss.ener_spin import EnergySpinLoss as EnergySpinLossDP
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class EnergySpinLoss(EnergySpinLossDP):
    pass
