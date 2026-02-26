# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3DP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("dpa3")
@torch_module
class DescrptDPA3(DescrptDPA3DP):
    pass
