# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

from deepmd.dpmodel.fitting.dpa4_ener import GLUFittingNet as GLUFittingNetDP
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMNetworkCollection as SeZMNetworkCollectionDP,
)
from deepmd.tf2.common import (
    register_dpmodel_mapping,
    tf2_module,
)
from deepmd.tf2.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.tf2.utils import network as _tf2_network  # noqa: F401


@tf2_module
class GLUFittingNet(GLUFittingNetDP):
    pass


register_dpmodel_mapping(
    GLUFittingNetDP,
    lambda v: GLUFittingNet.deserialize(v.serialize()),
)


@tf2_module
class SeZMNetworkCollection(SeZMNetworkCollectionDP):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }


register_dpmodel_mapping(
    SeZMNetworkCollectionDP,
    lambda v: SeZMNetworkCollection.deserialize(v.serialize()),
)


@BaseFitting.register("dpa4_ener")
@BaseFitting.register("sezm_ener")
@tf2_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)
