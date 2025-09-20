# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
)

from deepmd.dpmodel.model import (
    DPModelCommon,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.model.base_model import (
    forward_common_atomic,
)


def make_jax_dp_model_from_dpmodel(
    dpmodel_model: type[DPModelCommon], jax_atomicmodel: type[DPAtomicModel]
) -> type[DPModelCommon]:
    """Make a JAX backend DP model from a DPModel backend DP model.

    Parameters
    ----------
    dpmodel_model : type[DPModelCommon]
        The DPModel backend DP model.
    jax_atomicmodel : type[DPAtomicModel]
        The JAX backend DP atomic model.

    Returns
    -------
    type[DPModelCommon]
        The JAX backend DP model.
    """

    @flax_module
    class jax_model(dpmodel_model):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "atomic_model":
                value = jax_atomicmodel.deserialize(value.serialize())
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
            return dpmodel_model.format_nlist(
                self,
                jax.lax.stop_gradient(extended_coord),
                extended_atype,
                nlist,
                extra_nlist_sort=extra_nlist_sort,
            )

    return jax_model
