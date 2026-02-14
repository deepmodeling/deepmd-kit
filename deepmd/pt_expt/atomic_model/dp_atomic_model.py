# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel as DPAtomicModelDP
from deepmd.pt_expt.common import (
    dpmodel_setattr,
    register_dpmodel_mapping,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)


class DPAtomicModel(DPAtomicModelDP, torch.nn.Module):
    base_descriptor_cls = BaseDescriptor
    base_fitting_cls = BaseFitting

    def __init__(
        self, descriptor: Any, fitting: Any, *args: Any, **kwargs: Any
    ) -> None:
        torch.nn.Module.__init__(self)
        # Convert descriptor and fitting to pt_expt versions if they are dpmodel instances
        # The dpmodel_setattr mechanism will handle this automatically via registry
        from deepmd.pt_expt.common import (
            try_convert_module,
        )

        descriptor_pt = try_convert_module(descriptor)
        fitting_pt = try_convert_module(fitting)
        # If conversion failed (not registered), use original (assume already pt_expt)
        if descriptor_pt is None:
            descriptor_pt = descriptor
        if fitting_pt is None:
            fitting_pt = fitting
        DPAtomicModelDP.__init__(self, descriptor_pt, fitting_pt, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Ensure torch.nn.Module.__call__ drives forward() for export/tracing.
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        handled, value = dpmodel_setattr(self, name, value)
        if not handled:
            super().__setattr__(name, value)

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
