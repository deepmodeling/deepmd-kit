# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel.descriptor.se_t import DescrptSeT as DescrptSeTDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e3_expt")
@BaseDescriptor.register("se_at_expt")
@BaseDescriptor.register("se_a_3be_expt")
@torch_module
class DescrptSeT(DescrptSeTDP):
    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        descrpt, rot_mat, g2, h2, sw = self.call(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        return descrpt, rot_mat, g2, h2, sw
