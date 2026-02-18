# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.se_atten_v2 import DescrptSeAttenV2 as DescrptSeAttenV2DP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_atten_v2")
@torch_module
class DescrptSeAttenV2(DescrptSeAttenV2DP):
    pass
