# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd as DescrptBlockSeTTebdDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class DescrptBlockSeTTebd(DescrptBlockSeTTebdDP):
    pass


register_dpmodel_mapping(
    DescrptBlockSeTTebdDP,
    lambda v: DescrptBlockSeTTebd.deserialize(v.serialize()),
)
