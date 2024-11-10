# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_hessian_model import (
    make_hessian_model,
)
from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel)
DPEnergyModel_ = make_hessian_model(DPEnergyModel_)


@BaseModel.register("ener_hess")
class EnergyHessianModel(DPModelCommon, DPEnergyModel_):
    model_type = "ener_hess"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": deepcopy(out_def_data["energy"]),
            "energy": deepcopy(out_def_data["energy_redu"]),
        }
        if self.do_grad_r("energy"):
            output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
            output_def["atom_virial"].squeeze(-3)
        output_def["hessian"] = deepcopy(out_def_data["energy_derv_r_derv_r"])
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
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        self.requires_hessian("energy")
        hess = self._cal_hessian_all(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
        )
        model_ret.update(hess)
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(
                        -3
                    )
            else:
                model_predict["force"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
            model_predict["hessian"] = model_ret["energy_derv_r_derv_r"]
            model_predict["hessian"].squeeze(-2)
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict
