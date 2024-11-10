# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
)

from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    DPZBLLinearEnergyAtomicModel as DPZBLLinearEnergyAtomicModelDP,
)
from deepmd.jax.atomic_model.base_atomic_model import (
    base_atomic_model_set_attr,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.env import (
    jax,
    jnp,
)


@flax_module
class DPZBLLinearEnergyAtomicModel(DPZBLLinearEnergyAtomicModelDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = base_atomic_model_set_attr(name, value)
        if name == "mapping_list":
            value = [ArrayAPIVariable(to_jax_array(vv)) for vv in value]
        elif name == "zbl_weight":
            value = ArrayAPIVariable(to_jax_array(value))
        elif name == "models":
            value = [
                DPAtomicModel.deserialize(value[0].serialize()),
                PairTabAtomicModel.deserialize(value[1].serialize()),
            ]
        return super().__setattr__(name, value)

    def forward_common_atomic(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: Optional[jnp.ndarray] = None,
        fparam: Optional[jnp.ndarray] = None,
        aparam: Optional[jnp.ndarray] = None,
    ) -> dict[str, jnp.ndarray]:
        return super().forward_common_atomic(
            extended_coord,
            extended_atype,
            jax.lax.stop_gradient(nlist),
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
        )
