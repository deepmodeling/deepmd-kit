# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_atten")
@BaseDescriptor.register("dpa1")
@torch_module
class DescrptDPA1(DescrptDPA1DP):
    pass
