# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingNetDP
from deepmd.pt_expt.common import (
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("dos")
@torch_module
class DOSFittingNet(DOSFittingNetDP):
    pass
