# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.make_base_descriptor import (
    make_base_descriptor,
)
from deepmd.jax.env import (
    jnp,
)

BaseDescriptor = make_base_descriptor(jnp.ndarray)
