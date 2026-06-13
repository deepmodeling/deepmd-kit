# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.dpa1 import DescrptBlockSeAtten as DescrptBlockSeAttenDP
from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP
from deepmd.dpmodel.descriptor.dpa1 import GatedAttentionLayer as GatedAttentionLayerDP
from deepmd.dpmodel.descriptor.dpa1 import (
    NeighborGatedAttention as NeighborGatedAttentionDP,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    NeighborGatedAttentionLayer as NeighborGatedAttentionLayerDP,
)

from ..common import (
    array_api_strict_module,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401
from ..utils import type_embed as _strict_type_embed  # noqa: F401
from .base_descriptor import (
    BaseDescriptor,
)


@array_api_strict_module
class GatedAttentionLayer(GatedAttentionLayerDP):
    pass


@array_api_strict_module
class NeighborGatedAttentionLayer(NeighborGatedAttentionLayerDP):
    pass


@array_api_strict_module
class NeighborGatedAttention(NeighborGatedAttentionDP):
    pass


@array_api_strict_module
class DescrptBlockSeAtten(DescrptBlockSeAttenDP):
    pass


@BaseDescriptor.register("dpa1")
@BaseDescriptor.register("se_atten")
@array_api_strict_module
class DescrptDPA1(DescrptDPA1DP):
    pass
