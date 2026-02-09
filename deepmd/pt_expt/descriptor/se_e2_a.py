# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeAArrayAPI as DescrptSeADP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e2_a_expt")
@BaseDescriptor.register("se_a_expt")
@torch_module
class DescrptSeA(DescrptSeADP):
    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        extended_atype_embd: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        type_embedding: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del extended_atype_embd, type_embedding
        descrpt, rot_mat, g2, h2, sw = self.call(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        return descrpt, rot_mat, g2, h2, sw
