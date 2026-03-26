# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.loss.dos import DOSLoss as DOSLossDP
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class DOSLoss(DOSLossDP):
    pass
