# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPPropertyAtomicModel,
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

DPPropertyModel_ = make_model(DPPropertyAtomicModel)


@BaseModel.register("property")
class PropertyModel(DPModelCommon, DPPropertyModel_):
    model_type = "property"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPPropertyModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_property": deepcopy(out_def_data["property"]),
            "property": deepcopy(out_def_data["property_redu"]),
        }
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
        model_predict = {}
        model_predict["atom_property"] = model_ret["property"]
        model_predict["property"] = model_ret["property_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    @torch.jit.export
    def get_task_dim(self) -> int:
        """Get the output dimension of PropertyFittingNet."""
        return self.get_fitting_net().dim_out

    @torch.jit.export
    def get_intensive(self) -> bool:
        """Get whether the property is intensive."""
        return self.model_output_def()["property"].intensive

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
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
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
        model_predict = {}
        model_predict["atom_property"] = model_ret["property"]
        model_predict["property"] = model_ret["property_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict
