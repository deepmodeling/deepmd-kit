# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingNetDP
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP

from ..common import (
    to_array_api_strict_array,
)
from ..utils.exclude_mask import (
    AtomExcludeMask,
)
from ..utils.network import (
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
        value = to_array_api_strict_array(value)
    elif name == "emask":
        value = AtomExcludeMask(value.ntypes, value.exclude_types)
    elif name == "nets":
        value = NetworkCollection.deserialize(value.serialize())
    return value


class EnergyFittingNet(EnergyFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)


class DOSFittingNet(DOSFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)
