# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.fitting.dipole_fitting import DipoleFitting as DipoleFittingNetDP
from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingNetDP
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.dpmodel.fitting.polarizability_fitting import (
    PolarFitting as PolarFittingNetDP,
)
from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet as PropertyFittingNetDP,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)


@BaseFitting.register("ener")
@flax_module
class EnergyFittingNet(EnergyFittingNetDP):
    pass


@BaseFitting.register("property")
@flax_module
class PropertyFittingNet(PropertyFittingNetDP):
    pass


@BaseFitting.register("dos")
@flax_module
class DOSFittingNet(DOSFittingNetDP):
    pass


@BaseFitting.register("dipole")
@flax_module
class DipoleFittingNet(DipoleFittingNetDP):
    pass


@BaseFitting.register("polar")
@flax_module
class PolarFittingNet(PolarFittingNetDP):
    pass
