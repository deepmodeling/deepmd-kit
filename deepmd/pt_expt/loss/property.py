# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.loss.property import PropertyLoss as PropertyLossDP
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class PropertyLoss(PropertyLossDP):
    pass
