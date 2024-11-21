# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
)

from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel as DPAtomicModelDP
from deepmd.jax.atomic_model.base_atomic_model import (
    base_atomic_model_set_attr,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)


def make_jax_dp_atomic_model_from_dpmodel(
    dpmodel_atomic_model: type[DPAtomicModelDP],
) -> type[DPAtomicModelDP]:
    """Make a JAX backend DP atomic model from a DPModel backend DP atomic model.

    Parameters
    ----------
    dpmodel_atomic_model : type[DPAtomicModelDP]
        The DPModel backend DP atomic model.

    Returns
    -------
    type[DPAtomicModel]
        The JAX backend DP atomic model.
    """

    @flax_module
    class jax_atomic_model(dpmodel_atomic_model):
        base_descriptor_cls = BaseDescriptor
        """The base descriptor class."""
        base_fitting_cls = BaseFitting
        """The base fitting class."""

        def __setattr__(self, name: str, value: Any) -> None:
            value = base_atomic_model_set_attr(name, value)
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

    return jax_atomic_model


class DPAtomicModel(make_jax_dp_atomic_model_from_dpmodel(DPAtomicModelDP)):
    pass
