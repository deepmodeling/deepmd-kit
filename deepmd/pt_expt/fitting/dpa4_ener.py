# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

import torch

from deepmd.dpmodel.fitting.dpa4_ener import GLUFittingNet as GLUFittingNetDP
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMNetworkCollection as SeZMNetworkCollectionDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@torch_module
class GLUFittingNet(GLUFittingNetDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


register_dpmodel_mapping(
    GLUFittingNetDP,
    lambda v: GLUFittingNet.deserialize(v.serialize()),
)


@torch_module
class SeZMNetworkCollection(SeZMNetworkCollectionDP):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._module_networks = torch.nn.ModuleDict()
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: int | tuple | str, value: Any) -> None:
        super().__setitem__(key, value)
        idx = self._convert_key(key)
        net = self._networks[idx]
        key_str = str(idx)
        if isinstance(net, torch.nn.Module):
            self._module_networks[key_str] = net
        elif key_str in self._module_networks:
            del self._module_networks[key_str]


register_dpmodel_mapping(
    SeZMNetworkCollectionDP,
    lambda v: SeZMNetworkCollection.deserialize(v.serialize()),
)


@BaseFitting.register("dpa4_ener")
@BaseFitting.register("sezm_ener")
@torch_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)
