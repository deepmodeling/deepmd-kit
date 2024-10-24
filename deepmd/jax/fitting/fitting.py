# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingNetDP
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.utils.exclude_mask import (
    AtomExcludeMask,
)
from deepmd.jax.utils.network import (
    NetworkCollection,
)


def setattr_for_general_fitting(name: str, value: Any) -> Any:
    if name in {
        "bias_atom_e",
        "fparam_avg",
        "fparam_inv_std",
        "aparam_avg",
        "aparam_inv_std",
    }:
        value = to_jax_array(value)
        if value is not None:
            value = ArrayAPIVariable(value)
    elif name == "emask":
        value = AtomExcludeMask(value.ntypes, value.exclude_types)
    elif name == "nets":
        value = NetworkCollection.deserialize(value.serialize())
    return value


@BaseFitting.register("ener")
@flax_module
class EnergyFittingNet(EnergyFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)


@BaseFitting.register("dos")
@flax_module
class DOSFittingNet(DOSFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)
