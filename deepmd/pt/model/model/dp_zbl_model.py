# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.dpmodel.model.dp_model import (
    DPModel,
)
from deepmd.pt.model.atomic_model import (
    DPZBLLinearEnergyAtomicModel,
)
from deepmd.pt.model.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .make_model import (
    make_model,
)

DPZBLModel_ = make_model(DPZBLLinearEnergyAtomicModel)


@BaseModel.register("zbl")
class DPZBLModel(DPZBLModel_):
    model_type = "ener"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
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
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        else:
            model_predict["force"] = model_ret["dforce"]
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
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -3
                )
        else:
            assert model_ret["dforce"] is not None
            model_predict["dforce"] = model_ret["dforce"]
        return model_predict

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["dpmodel"] = DPModel.update_sel(
            global_jdata, local_jdata["dpmodel"]
        )
        return local_jdata_cpy

    @classmethod
    def deserialize(cls, data) -> "DPZBLModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        sw_rmin = data.pop("sw_rmin")
        sw_rmax = data.pop("sw_rmax")
        smin_alpha = data.pop("smin_alpha")

        [dp_model, zbl_model], type_map = LinearEnergyAtomicModel.deserialize(
            data.pop("models")
        )

        data.pop("@class", None)
        data.pop("type", None)
        return cls(
            dp_model=dp_model,
            zbl_model=zbl_model,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
            type_map=type_map,
            smin_alpha=smin_alpha,
            **data,
        )
