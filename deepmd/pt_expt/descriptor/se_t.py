# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.se_t import DescrptSeT as DescrptSeTDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("se_e3")
@BaseDescriptor.register("se_at")
@BaseDescriptor.register("se_a_3be")
@torch_module
class DescrptSeT(DescrptSeTDP):
    _update_sel_cls = UpdateSel
