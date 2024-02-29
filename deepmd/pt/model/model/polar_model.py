# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.utils.data import (
    DataRequirementItem,
)

from .dp_model import (
    DPModel,
)


class PolarModel(DPModel):
    model_type = "polar"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.fitting_net is not None:
            model_predict = {}
            model_predict["polar"] = model_ret["polar"]
            model_predict["global_polar"] = model_ret["polar_redu"]
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.fitting_net is not None:
            model_predict = {}
            model_predict["polar"] = model_ret["polar"]
            model_predict["global_polar"] = model_ret["polar_redu"]
        else:
            model_predict = model_ret
        return model_predict

    @property
    def get_data_requirement(self) -> List[DataRequirementItem]:
        data_requirement = [
            DataRequirementItem(
                "polar",
                ndof=9,
                atomic=False,
                must=False,
                high_prec=False,
                type_sel=self.get_sel_type(),
            ),
            DataRequirementItem(
                "atomic_polar",
                ndof=9,
                atomic=True,
                must=False,
                high_prec=False,
                type_sel=self.get_sel_type(),
            ),
        ]
        return data_requirement
