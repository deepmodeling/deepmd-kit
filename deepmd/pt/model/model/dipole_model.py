# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from .dp_model import (
    DPModel,
)


class DipoleModel(DPModel):
    model_type = "dipole"

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
            do_atomic_virial=False,
        )
        if self.fitting_net is not None:
            model_predict = {}
            model_predict["atom_dipole"] = model_ret["dipole"]
            model_predict["dipole"] = model_ret["dipole_redu"]
            if self.do_grad("dipole"):
                model_predict["dipole_grad_r"] = model_ret["dipole_derv_r"].squeeze(-2)
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

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
            do_atomic_virial=False,
        )
        if self.fitting_net is not None:
            model_predict = {}
            model_predict["atom_dipole"] = model_ret["dipole"]
            model_predict["energy"] = model_ret["energy_redu"]
        else:
            model_predict = model_ret
        return model_predict
