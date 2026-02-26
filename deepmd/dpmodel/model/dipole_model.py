# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model import (
    DPDipoleAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPDipoleModel_ = make_model(DPDipoleAtomicModel)


@BaseModel.register("dipole")
class DipoleModel(DPModelCommon, DPDipoleModel_):
    model_type = "dipole"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPDipoleModel_.__init__(self, *args, **kwargs)

    def call(
        self,
        coord: Array,
        atype: Array,
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
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

    def call_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
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
