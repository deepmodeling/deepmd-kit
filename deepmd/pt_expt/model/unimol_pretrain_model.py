# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.atomic_model import (
    DPUniMolAtomicModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)

DPUniMolPretrainModel_ = make_model(DPUniMolAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("unimol_pretrain")
class UniMolPretrainModel(DPModelCommon, DPUniMolPretrainModel_):
    """Uni-Mol v1 self-supervised pretraining on the PyTorch-Exportable backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        DPUniMolPretrainModel_.__init__(self, *args, **kwargs)

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the pretraining heads on a frame."""
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        return {k: v for k, v in model_ret.items() if v is not None}

    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the pretraining heads on an extended frame."""
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            charge_spin=charge_spin,
        )
        return {k: v for k, v in model_ret.items() if v is not None}

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        """The head outputs, passed through under their own names."""
        return dict(self.model_output_def().get_data())
