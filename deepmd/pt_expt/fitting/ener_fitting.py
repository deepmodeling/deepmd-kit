# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("ener")
@torch_module
class EnergyFittingNet(EnergyFittingNetDP):
    """Energy fitting net for pt_expt backend.

    This inherits from dpmodel EnergyFittingNet to get the correct serialize() method.
    """

    pass


register_dpmodel_mapping(
    EnergyFittingNetDP,
    lambda v: EnergyFittingNet.deserialize(v.serialize()),
)
