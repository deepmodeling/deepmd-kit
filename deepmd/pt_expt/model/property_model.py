# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.atomic_model import (
    DPPropertyAtomicModel,
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

DPPropertyModel_ = make_model(DPPropertyAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("property")
class PropertyModel(DPModelCommon, DPPropertyModel_):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPPropertyModel_.__init__(self, *args, **kwargs)

    def get_var_name(self) -> str:
        """Get the name of the property."""
        return self.get_fitting_net().var_name

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
        var_name = self.get_var_name()
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
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
        var_name = self.get_var_name()
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        var_name = self.get_var_name()
        output_def = {
            f"atom_{var_name}": out_def_data[var_name],
            var_name: out_def_data[f"{var_name}_redu"],
        }
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
