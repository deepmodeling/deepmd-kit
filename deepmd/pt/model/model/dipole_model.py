# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPDipoleAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPDOSModel_ = make_model(DPDipoleAtomicModel)


@BaseModel.register("dipole")
class DipoleModel(DPModelCommon, DPDOSModel_):
    model_type = "dipole"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPDOSModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "dipole": deepcopy(out_def_data["dipole"]),
            "global_dipole": deepcopy(out_def_data["dipole_redu"]),
        }
        if self.do_grad_r("dipole"):
            output_def["force"] = deepcopy(out_def_data["dipole_derv_r"])
            output_def["force"].squeeze(-2)
        if self.do_grad_c("dipole"):
            output_def["virial"] = deepcopy(out_def_data["dipole_derv_c_redu"])
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = deepcopy(out_def_data["dipole_derv_c"])
            output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = deepcopy(out_def_data["mask"])
        return output_def

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
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["dipole"] = model_ret["dipole"]
            model_predict["global_dipole"] = model_ret["dipole_redu"]
            if self.do_grad_r("dipole"):
                model_predict["force"] = model_ret["dipole_derv_r"].squeeze(-2)
            if self.do_grad_c("dipole"):
                model_predict["virial"] = model_ret["dipole_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["dipole_derv_c"].squeeze(
                        -3
                    )
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
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
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["dipole"] = model_ret["dipole"]
            model_predict["global_dipole"] = model_ret["dipole_redu"]
            if self.do_grad_r("dipole"):
                model_predict["extended_force"] = model_ret["dipole_derv_r"].squeeze(-2)
            if self.do_grad_c("dipole"):
                model_predict["virial"] = model_ret["dipole_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "dipole_derv_c"
                    ].squeeze(-3)
        else:
            model_predict = model_ret
        return model_predict
