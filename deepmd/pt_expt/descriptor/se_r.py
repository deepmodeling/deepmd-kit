# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.se_r import DescrptSeR as DescrptSeRDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e2_r")
@BaseDescriptor.register("se_r")
@torch_module
class DescrptSeR(DescrptSeRDP):
    pass
