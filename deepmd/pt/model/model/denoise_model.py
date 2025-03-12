# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPDenoiseAtomicModel,
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

DPDenoiseModel_ = make_model(DPDenoiseAtomicModel)


@BaseModel.register("denoise")
class DenoiseModel(DPModelCommon, DPDenoiseModel_):
    model_type = "property"

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        DPModelCommon.__init__(self)
        DPDenoiseModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        pass
        """
        out_def_data = self.model_output_def().get_data()
        output_def = {
            f"atom_{self.get_var_name()}": out_def_data[self.get_var_name()],
            self.get_var_name(): out_def_data[f"{self.get_var_name()}_redu"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def
        """

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
        model_predict["updated_coord"] = model_ret["updated_coord"]
        model_predict["atom_strain_components"] = model_ret["strain_components"]
        model_predict["strain_components"] = model_ret["strain_components_redu"]
        model_predict["logits"] = model_ret["logits"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
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
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        #TODO: implement forward_lower
        pass
