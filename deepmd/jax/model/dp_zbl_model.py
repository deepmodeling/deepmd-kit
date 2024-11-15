# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
)

from deepmd.dpmodel.model.dp_zbl_model import DPZBLModel as DPZBLModelDP
from deepmd.jax.atomic_model.linear_atomic_model import (
    DPZBLLinearEnergyAtomicModel,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.model.base_model import (
    BaseModel,
    forward_common_atomic,
)


@BaseModel.register("zbl")
@flax_module
class DPZBLModel(DPZBLModelDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "atomic_model":
            value = DPZBLLinearEnergyAtomicModel.deserialize(value.serialize())
        return super().__setattr__(name, value)

    def forward_common_atomic(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: Optional[jnp.ndarray] = None,
        fparam: Optional[jnp.ndarray] = None,
        aparam: Optional[jnp.ndarray] = None,
        do_atomic_virial: bool = False,
    ):
        return forward_common_atomic(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

    def format_nlist(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        extra_nlist_sort: bool = False,
    ):
        return DPZBLModelDP.format_nlist(
            self,
            jax.lax.stop_gradient(extended_coord),
            extended_atype,
            nlist,
            extra_nlist_sort=extra_nlist_sort,
        )
