# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
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
        model_predict["atom_property"] = model_ret["property"]
        # TODO:
        natoms = model_predict["atom_property"].shape[1]
        model_predict["property"] = model_ret["property_redu"] / natoms 
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
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
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
        model_predict = {}
        model_predict["atom_property"] = model_ret["property"]
        natoms = model_predict["atom_property"].shape[1]
        model_predict["property"] = model_ret["property_redu"] / natoms
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict
