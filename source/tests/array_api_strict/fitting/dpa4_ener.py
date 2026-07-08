# SPDX-License-Identifier: LGPL-3.0-or-later
from importlib import (
    import_module,
)
from typing import (
    ClassVar,
)

from deepmd.dpmodel.fitting.dpa4_ener import GLUFittingNet as GLUFittingNetDP
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMNetworkCollection as SeZMNetworkCollectionDP,
)

from ..common import (
    array_api_strict_module,
)

import_module("..utils.network", __package__)


@array_api_strict_module
class GLUFittingNet(GLUFittingNetDP):
    pass


@array_api_strict_module
class SeZMNetworkCollection(SeZMNetworkCollectionDP):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }


@array_api_strict_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    pass
