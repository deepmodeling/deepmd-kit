# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPDensityAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_density_model import (
    make_density_model,
)

DPDensityModel_ = make_density_model(DPDensityAtomicModel)


@BaseModel.register("grid_density")
class GridDensityModel(DPModelCommon, DPDensityModel_):
    model_type = "grid_density"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPDensityModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "density": deepcopy(out_def_data["density"]),
        }
        if "mask" in out_def_data:
            output_def["mask"] = deepcopy(out_def_data["mask"])
        return output_def

    @torch.jit.export
    def has_grid(self) -> bool:
        """Returns whether it has grid input and output."""
        return True

    def forward(
        self,
        coord,
        atype,
        grid,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            grid,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["density"] = model_ret["density"]
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
        raise NotImplementedError