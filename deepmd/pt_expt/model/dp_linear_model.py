# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)

DPLinearModel_ = make_model(LinearEnergyAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("linear_ener")
class LinearEnergyModel(DPModelCommon, DPLinearModel_):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPLinearModel_.__init__(self, *args, **kwargs)

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
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy") and model_ret["energy_derv_r"] is not None:
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy") and model_ret["energy_derv_c_redu"] is not None:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret["energy_derv_c"] is not None:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
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
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy") and model_ret.get("energy_derv_r") is not None:
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy") and model_ret.get("energy_derv_c_redu") is not None:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret.get("energy_derv_c") is not None:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -2
                )
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
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

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        type_map = local_jdata_cpy["type_map"]
        min_nbor_dist = None
        for idx, sub_model in enumerate(local_jdata_cpy["models"]):
            if "tab_file" not in sub_model:
                sub_model, temp_min = DPModelCommon.update_sel(
                    train_data, type_map, local_jdata_cpy["models"][idx]
                )
                local_jdata_cpy["models"][idx] = sub_model
                if min_nbor_dist is None or temp_min <= min_nbor_dist:
                    min_nbor_dist = temp_min
        return local_jdata_cpy, min_nbor_dist
