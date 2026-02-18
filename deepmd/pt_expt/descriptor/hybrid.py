# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("hybrid")
@torch_module
class DescrptHybrid(DescrptHybridDP):
    pass
