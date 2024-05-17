# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPEnergyAtomicModel,
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

DPEnergyModel_ = make_model(DPEnergyAtomicModel)


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    model_type = "ener"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
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

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
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
