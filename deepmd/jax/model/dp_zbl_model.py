# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.model.dp_zbl_model import DPZBLModel as DPZBLModelDP
from deepmd.jax.atomic_model.linear_atomic_model import (
    DPZBLLinearEnergyAtomicModel as _DPZBLLinearEnergyAtomicModel,
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
from deepmd.jax.model.dp_model import (
    validate_neighbor_contract,
)


@BaseModel.register("zbl")
@flax_module
class DPZBLModel(DPZBLModelDP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        validate_neighbor_contract(self.atomic_model)

    @staticmethod
    def uses_full_neighbor_list() -> bool:
        """The switched tabulation model uses a bounded neighbor list."""
        return False

    @classmethod
    def deserialize(cls, data: dict) -> "DPZBLModel":
        """Restore composed children with JAX parameter and adapter state."""
        return cls(atomic_model_=_DPZBLLinearEnergyAtomicModel.deserialize(data))

    def forward_common_atomic(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        do_atomic_virial: bool = False,
        extended_coord_corr: jnp.ndarray | None = None,
        comm_dict: dict | None = None,
        charge_spin: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        del comm_dict  # JAX path has no MPI ghost exchange
        return forward_common_atomic(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            extended_coord_corr=extended_coord_corr,
            charge_spin=charge_spin,
        )

    def format_nlist(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        extra_nlist_sort: bool = False,
    ) -> jnp.ndarray:
        return DPZBLModelDP.format_nlist(
            self,
            jax.lax.stop_gradient(extended_coord),
            extended_atype,
            nlist,
            extra_nlist_sort=extra_nlist_sort,
        )
