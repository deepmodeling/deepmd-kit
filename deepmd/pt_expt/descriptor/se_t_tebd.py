# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e3_tebd")
@torch_module
class DescrptSeTTebd(DescrptSeTTebdDP):
    pass
