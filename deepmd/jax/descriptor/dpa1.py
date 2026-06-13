# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
import deepmd.jax.utils.type_embed as _jax_type_embed  # noqa: F401
from deepmd.dpmodel.descriptor.dpa1 import DescrptBlockSeAtten as DescrptBlockSeAttenDP
from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP
from deepmd.dpmodel.descriptor.dpa1 import GatedAttentionLayer as GatedAttentionLayerDP
from deepmd.dpmodel.descriptor.dpa1 import (
    NeighborGatedAttention as NeighborGatedAttentionDP,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    NeighborGatedAttentionLayer as NeighborGatedAttentionLayerDP,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)


@flax_module
class GatedAttentionLayer(GatedAttentionLayerDP):
    pass


@flax_module
class NeighborGatedAttentionLayer(NeighborGatedAttentionLayerDP):
    pass


@flax_module
class NeighborGatedAttention(NeighborGatedAttentionDP):
    pass


@flax_module
class DescrptBlockSeAtten(DescrptBlockSeAttenDP):
    pass


@BaseDescriptor.register("dpa1")
@BaseDescriptor.register("se_atten")
@flax_module
class DescrptDPA1(DescrptDPA1DP):
    pass
