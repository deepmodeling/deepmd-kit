# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.descriptor.dpa1 as _jax_dpa1  # noqa: F401
import deepmd.jax.descriptor.dpa2 as _jax_dpa2  # noqa: F401
import deepmd.jax.descriptor.dpa3 as _jax_dpa3  # noqa: F401
import deepmd.jax.descriptor.se_atten_v2 as _jax_se_atten_v2  # noqa: F401
import deepmd.jax.descriptor.se_e2_a as _jax_se_e2_a  # noqa: F401
import deepmd.jax.descriptor.se_e2_r as _jax_se_e2_r  # noqa: F401
import deepmd.jax.descriptor.se_t as _jax_se_t  # noqa: F401
import deepmd.jax.descriptor.se_t_tebd as _jax_se_t_tebd  # noqa: F401
from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("hybrid")
@flax_module
class DescrptHybrid(DescrptHybridDP):
    pass
