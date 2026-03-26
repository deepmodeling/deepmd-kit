# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.loss.tensor import TensorLoss as TensorLossDP
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class TensorLoss(TensorLossDP):
    pass
