# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPDOSAtomicModel,
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

DPDOSModel_ = make_model(DPDOSAtomicModel)


@BaseModel.register("dos")
class DOSModel(DPModelCommon, DPDOSModel_):
    model_type = "dos"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPDOSModel_.__init__(self, *args, **kwargs)

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
            model_predict["atom_dos"] = model_ret["dos"]
            model_predict["dos"] = model_ret["dos_redu"]

            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    @torch.jit.export
    def get_numb_dos(self) -> int:
        """Get the number of  DOS for DOSFittingNet."""
        return self.get_fitting_net().dim_out

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
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_dos"] = model_ret["dos"]
            model_predict["dos"] = model_ret["dos_redu"]

        else:
            model_predict = model_ret
        return model_predict
