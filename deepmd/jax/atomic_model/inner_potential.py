# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX state wrappers for analytical short-range pair potentials."""

from deepmd.dpmodel.atomic_model.inner_potential import (
    InnerPotential as InnerPotentialDP,
)
from deepmd.dpmodel.atomic_model.inner_potential import (
    InnerPotentialAtomicModel as InnerPotentialAtomicModelDP,
)
from deepmd.jax.common import (
    flax_module,
    register_dpmodel_mapping,
)


@flax_module
class InnerPotential(InnerPotentialDP):
    pass


register_dpmodel_mapping(
    InnerPotentialDP,
    lambda value: InnerPotential(value.type_map, value.mode, value.truncation),
)


@flax_module
class InnerPotentialAtomicModel(InnerPotentialAtomicModelDP):
    pass
