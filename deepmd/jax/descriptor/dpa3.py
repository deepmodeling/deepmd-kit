# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.descriptor.repflows as _jax_repflows  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
import deepmd.jax.utils.type_embed as _jax_type_embed  # noqa: F401
from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3DP
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("dpa3")
@flax_module
class DescrptDPA3(DescrptDPA3DP):
    pass
