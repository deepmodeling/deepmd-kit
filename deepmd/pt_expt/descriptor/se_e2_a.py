# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeAArrayAPI as DescrptSeADP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e2_a")
@BaseDescriptor.register("se_a")
@torch_module
class DescrptSeA(DescrptSeADP):
    pass
