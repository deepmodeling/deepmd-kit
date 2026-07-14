# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import deepmd.jax.atomic_model.dp_atomic_model as _jax_dp_atomic_model  # noqa: F401
import deepmd.jax.atomic_model.pairtab_atomic_model as _jax_pairtab_model  # noqa: F401
import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    DPZBLLinearEnergyAtomicModel as DPZBLLinearEnergyAtomicModelDP,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.env import (
    jax,
    jnp,
)


@flax_module
class DPZBLLinearEnergyAtomicModel(DPZBLLinearEnergyAtomicModelDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "zbl_weight":
            # discard since it's only used in tests
            # to fix flax.errors.TraceContextError: Cannot mutate 'FlaxModule' from different trace level
            return None
        return super().__setattr__(name, value)

    def forward_common_atomic(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        comm_dict: dict | None = None,
        charge_spin: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        del comm_dict  # JAX path has no MPI ghost exchange
        return super().forward_common_atomic(
            extended_coord,
            extended_atype,
            jax.lax.stop_gradient(nlist),
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
        )
