# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.invar_fitting import InvarFitting as InvarFittingDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)


@BaseFitting.register("invar")
@torch_module
class InvarFitting(InvarFittingDP):
    pass
