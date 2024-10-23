# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.make_base_fitting import (
    make_base_fitting,
)
from deepmd.jax.env import (
    jnp,
)

BaseFitting = make_base_fitting(jnp.ndarray)
