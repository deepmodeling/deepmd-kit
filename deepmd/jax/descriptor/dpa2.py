# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.descriptor.dpa1 as _jax_dpa1  # noqa: F401
import deepmd.jax.descriptor.repformers as _jax_repformers  # noqa: F401
import deepmd.jax.descriptor.se_t_tebd as _jax_se_t_tebd  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
import deepmd.jax.utils.type_embed as _jax_type_embed  # noqa: F401
from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("dpa2")
@flax_module
class DescrptDPA2(DescrptDPA2DP):
    pass
