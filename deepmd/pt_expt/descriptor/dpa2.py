# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("dpa2")
@torch_module
class DescrptDPA2(DescrptDPA2DP):
    pass
