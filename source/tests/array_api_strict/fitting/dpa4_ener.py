# SPDX-License-Identifier: LGPL-3.0-or-later
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
    register_dpmodel_mapping,
)
from ..utils import network as _strict_network  # noqa: F401


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


register_dpmodel_mapping(
    GLUFittingNetDP,
    lambda v: GLUFittingNet.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    SeZMNetworkCollectionDP,
    lambda v: SeZMNetworkCollection.deserialize(v.serialize()),
)
