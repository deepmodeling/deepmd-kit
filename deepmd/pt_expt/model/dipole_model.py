# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.atomic_model import (
    DPDipoleAtomicModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)

DPDipoleModel_ = make_model(DPDipoleAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("dipole")
class DipoleModel(DPModelCommon, DPDipoleModel_):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPDipoleModel_.__init__(self, *args, **kwargs)

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["dipole"] = model_ret["dipole"]
        model_predict["global_dipole"] = model_ret["dipole_redu"]
        if self.do_grad_r("dipole") and model_ret["dipole_derv_r"] is not None:
            model_predict["force"] = model_ret["dipole_derv_r"]
        if self.do_grad_c("dipole") and model_ret["dipole_derv_c_redu"] is not None:
            model_predict["virial"] = model_ret["dipole_derv_c_redu"]
            if do_atomic_virial and model_ret["dipole_derv_c"] is not None:
                model_predict["atom_virial"] = model_ret["dipole_derv_c"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["dipole"] = model_ret["dipole"]
        model_predict["global_dipole"] = model_ret["dipole_redu"]
        if self.do_grad_r("dipole") and model_ret.get("dipole_derv_r") is not None:
            model_predict["extended_force"] = model_ret["dipole_derv_r"]
        if self.do_grad_c("dipole") and model_ret.get("dipole_derv_c_redu") is not None:
            model_predict["virial"] = model_ret["dipole_derv_c_redu"]
            if do_atomic_virial and model_ret.get("dipole_derv_c") is not None:
                model_predict["extended_virial"] = model_ret["dipole_derv_c"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "dipole": out_def_data["dipole"],
            "global_dipole": out_def_data["dipole_redu"],
        }
        if self.do_grad_r("dipole"):
            output_def["force"] = out_def_data["dipole_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("dipole"):
            output_def["virial"] = out_def_data["dipole_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["dipole_derv_c"]
            output_def["atom_virial"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> torch.nn.Module:
        model = self

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            extended_coord = extended_coord.detach().requires_grad_(True)
            return model.forward_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )

        return make_fx(fn)(
            extended_coord, extended_atype, nlist, mapping, fparam, aparam
        )
