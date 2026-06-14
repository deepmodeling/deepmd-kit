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
    array_api_strict_module,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401


@array_api_strict_module
class EnergyFittingNet(EnergyFittingNetDP):
    pass


@array_api_strict_module
class PropertyFittingNet(PropertyFittingNetDP):
    pass


@array_api_strict_module
class DOSFittingNet(DOSFittingNetDP):
    pass


@array_api_strict_module
class DipoleFittingNet(DipoleFittingNetDP):
    pass


@array_api_strict_module
class PolarFittingNet(PolarFittingNetDP):
    pass
