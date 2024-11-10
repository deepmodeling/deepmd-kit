# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.se_atten_v2 import DescrptSeAttenV2 as DescrptSeAttenV2DP
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.descriptor.dpa1 import (
    DescrptDPA1,
)


@BaseDescriptor.register("se_atten_v2")
class DescrptSeAttenV2(DescrptDPA1, DescrptSeAttenV2DP):
    pass
