# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model import (
    DPXASAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
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

DPXASModel_ = make_model(DPXASAtomicModel, T_Bases=(NativeOP, BaseModel))


@BaseModel.register("xas")
class XASModel(DPModelCommon, DPXASModel_):
    model_type = "xas"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPXASModel_.__init__(self, *args, **kwargs)

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
        model_predict["atom_xas"] = model_ret["xas"]
        model_predict["xas"] = model_ret["xas_redu"]
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
        model_predict["atom_xas"] = model_ret["xas"]
        model_predict["xas"] = model_ret["xas_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_xas": out_def_data["xas"],
            "xas": out_def_data["xas_redu"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def
