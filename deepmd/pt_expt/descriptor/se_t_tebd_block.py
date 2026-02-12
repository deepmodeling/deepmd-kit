# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd as DescrptBlockSeTTebdDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class DescrptBlockSeTTebd(DescrptBlockSeTTebdDP):
    def forward(
        self,
        nlist: torch.Tensor,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        atype_embd_ext: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        type_embedding: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return self.call(
            nlist,
            coord_ext,
            atype_ext,
            atype_embd_ext=atype_embd_ext,
            mapping=mapping,
            type_embedding=type_embedding,
        )


register_dpmodel_mapping(
    DescrptBlockSeTTebdDP,
    lambda v: DescrptBlockSeTTebd.deserialize(v.serialize()),
)
