# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX wrappers for the DPA4 gated energy fitting networks."""

from typing import (
    Any,
    ClassVar,
)

from deepmd.dpmodel.fitting.dpa4_ener import (
    CaseFiLMConditioner as CaseFiLMConditionerDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import GLUFittingNet as GLUFittingNetDP
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMNetworkCollection as SeZMNetworkCollectionDP,
)
from deepmd.dpmodel.utils.expert_route import (
    reject_moe,
)
from deepmd.jax.common import (
    flax_module,
    to_jax_array,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


@flax_module
class CaseFiLMConditioner(CaseFiLMConditionerDP):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._promote_parameters()

    def _promote_parameters(self) -> None:
        if not self.trainable:
            return
        for name in self.PARAMETER_ATTRIBUTES:
            value = getattr(self, name)
            if not isinstance(value, ArrayAPIParam):
                setattr(self, name, ArrayAPIParam(to_jax_array(value)))
        for name in self.PARAMETER_LIST_ATTRIBUTES:
            values = [
                value
                if isinstance(value, ArrayAPIParam)
                else ArrayAPIParam(to_jax_array(value))
                for value in getattr(self, name)
            ]
            setattr(self, name, values)

    def set_variables(self, variables: dict[str, Any]) -> None:
        super().set_variables(variables)
        self._promote_parameters()


@flax_module
class GLUFittingNet(GLUFittingNetDP):
    pass


@flax_module
class SeZMNetworkCollection(SeZMNetworkCollectionDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {"_networks"}
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }


@BaseFitting.register("dpa4_ener")
@BaseFitting.register("sezm_ener")
@flax_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        reject_moe(kwargs, "fitting network")
        super().__init__(*args, **kwargs)
