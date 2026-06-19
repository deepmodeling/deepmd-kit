# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
import deepmd.jax.utils.type_embed as _jax_type_embed  # noqa: F401
from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd as DescrptBlockSeTTebdDP,
)
from deepmd.dpmodel.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdDP
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)


@flax_module
class DescrptBlockSeTTebd(DescrptBlockSeTTebdDP):
    pass


@BaseDescriptor.register("se_e3_tebd")
@flax_module
class DescrptSeTTebd(DescrptSeTTebdDP):
    pass
