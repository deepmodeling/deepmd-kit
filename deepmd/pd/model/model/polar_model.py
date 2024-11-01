# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Optional,
)

import paddle

from deepmd.pd.model.atomic_model import (
    DPPolarAtomicModel,
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

DPDOSModel_ = make_model(DPPolarAtomicModel)


@BaseModel.register("polar")
class PolarModel(DPModelCommon, DPDOSModel_):
    model_type = "polar"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPDOSModel_.__init__(self, *args, **kwargs)
        DPModelCommon.__init__(self)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "polar": deepcopy(out_def_data["polarizability"]),
            "global_polar": deepcopy(out_def_data["polarizability_redu"]),
        }
        if "mask" in out_def_data:
            output_def["mask"] = deepcopy(out_def_data["mask"])
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
            model_predict["polar"] = model_ret["polarizability"]
            model_predict["global_polar"] = model_ret["polarizability_redu"]
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
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["polar"] = model_ret["polarizability"]
            model_predict["global_polar"] = model_ret["polarizability_redu"]
        else:
            model_predict = model_ret
        return model_predict
