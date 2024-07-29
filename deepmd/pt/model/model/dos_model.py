# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from .dp_model import (
    DPModel,
)


class DOSModel(DPModel):
    model_type = "dos"

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
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
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
            model_predict["atom_dos"] = model_ret["dos"]
            model_predict["dos"] = model_ret["dos_redu"]

        else:
            model_predict = model_ret
        return model_predict
