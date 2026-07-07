# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.fitting.dpa4_ener import GLUFittingNet as GLUFittingNetDP
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMNetworkCollection as SeZMNetworkCollectionDP,
)
from deepmd.jax.common import (
    flax_module,
    register_dpmodel_mapping,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)


@flax_module
class GLUFittingNet(GLUFittingNetDP):
    pass


register_dpmodel_mapping(
    GLUFittingNetDP,
    lambda v: GLUFittingNet.deserialize(v.serialize()),
)


@flax_module
class SeZMNetworkCollection(SeZMNetworkCollectionDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {"_networks", "networks"}
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }


register_dpmodel_mapping(
    SeZMNetworkCollectionDP,
    lambda v: SeZMNetworkCollection.deserialize(v.serialize()),
)


@BaseFitting.register("dpa4_ener")
@BaseFitting.register("sezm_ener")
@flax_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)
