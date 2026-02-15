# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel as DPAtomicModelDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)


@torch_module
class DPAtomicModel(DPAtomicModelDP):
    base_descriptor_cls = BaseDescriptor
    base_fitting_cls = BaseFitting

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
        )


register_dpmodel_mapping(
    DPAtomicModelDP,
    lambda v: DPAtomicModel.deserialize(v.serialize()),
)
