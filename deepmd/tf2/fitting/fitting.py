# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.dipole_fitting import DipoleFitting as DipoleFittingNetDP
from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingNetDP
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.dpmodel.fitting.polarizability_fitting import (
    PolarFitting as PolarFittingNetDP,
)
from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet as PropertyFittingNetDP,
)

from ..common import (
    tf2_module,
)
from ..utils import exclude_mask as _tf2_exclude_mask  # noqa: F401
from ..utils import network as _tf2_network  # noqa: F401
from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("ener")
@tf2_module
class EnergyFittingNet(EnergyFittingNetDP):
    pass


@BaseFitting.register("property")
@tf2_module
class PropertyFittingNet(PropertyFittingNetDP):
    pass


@BaseFitting.register("dos")
@tf2_module
class DOSFittingNet(DOSFittingNetDP):
    pass


@BaseFitting.register("dipole")
@tf2_module
class DipoleFittingNet(DipoleFittingNetDP):
    pass


@BaseFitting.register("polar")
@tf2_module
class PolarFittingNet(PolarFittingNetDP):
    pass
