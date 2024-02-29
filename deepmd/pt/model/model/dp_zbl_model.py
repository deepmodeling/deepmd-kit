# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPZBLLinearAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .make_model import (
    make_model,
)

DPZBLModel_ = make_model(DPZBLLinearAtomicModel)


@BaseModel.register("zbl")
class DPZBLModel(DPZBLModel_, BaseModel):
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
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        else:
            assert model_ret["dforce"] is not None
            model_predict["dforce"] = model_ret["dforce"]
        model_predict = model_ret
        return model_predict

    @property
    def data_requirement(self) -> List[DataRequirementItem]:
        data_requirement = [
            DataRequirementItem(
                "energy",
                ndof=1,
                atomic=False,
                must=False,
                high_prec=True,
            ),
            DataRequirementItem(
                "force",
                ndof=3,
                atomic=True,
                must=False,
                high_prec=False,
            ),
            DataRequirementItem(
                "virial",
                ndof=9,
                atomic=False,
                must=False,
                high_prec=False,
            ),
            DataRequirementItem(
                "atom_ener",
                ndof=1,
                atomic=True,
                must=False,
                high_prec=False,
            ),
            DataRequirementItem(
                "atom_pref",
                ndof=1,
                atomic=True,
                must=False,
                high_prec=False,
                repeat=3,
            ),
        ]
        return data_requirement
