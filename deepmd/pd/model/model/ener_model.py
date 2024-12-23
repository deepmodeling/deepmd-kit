# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import paddle

from deepmd.pd.model.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.pd.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel)


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    model_type = "ener"

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward(
        self,
        coord,
        atype,
        box: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, paddle.Tensor]:
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
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, paddle.Tensor]] = None,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-3)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
        else:
            model_predict = model_ret
        return model_predict
