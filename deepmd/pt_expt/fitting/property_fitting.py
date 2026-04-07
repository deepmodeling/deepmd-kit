# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet as PropertyFittingNetDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("property")
@torch_module
class PropertyFittingNet(PropertyFittingNetDP):
    pass
