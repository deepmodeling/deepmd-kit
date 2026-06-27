# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model.dp_zbl_model import DPZBLModel as DPZBLModelDP
from deepmd.tf2.atomic_model.linear_atomic_model import (  # noqa: F401
    DPZBLLinearEnergyAtomicModel as _DPZBLLinearEnergyAtomicModel,
)
from deepmd.tf2.common import (
    tf2_module,
)
from deepmd.tf2.env import (
    jnp,
    stop_gradient,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
    forward_common_atomic,
)


@BaseModel.register("zbl")
@tf2_module
class DPZBLModel(DPZBLModelDP):
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
        del comm_dict  # tf2 path has no MPI ghost exchange
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
            stop_gradient(extended_coord),
            extended_atype,
            nlist,
            extra_nlist_sort=extra_nlist_sort,
        )
