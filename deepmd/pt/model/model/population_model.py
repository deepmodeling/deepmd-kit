# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)
from deepmd.pt.model.atomic_model import (
    DPPopulationAtomicModel,
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

DPPopulationAtomicModel_ = make_model(DPPopulationAtomicModel)


@BaseModel.register("population")
class PopulationModel(DPModelCommon, DPPopulationAtomicModel_):
    """Model for fitting atomic charge population.

    Predicts per-atom alpha and beta spin channel electron populations.
    """

    model_type = "population"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the PopulationModel."""
        DPModelCommon.__init__(self)
        DPPopulationAtomicModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        """Return the output variable definitions exposed by this model."""
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "population": out_def_data["population"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the atomic charge population for the input structure."""
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        model_predict = {}
        model_predict["population"] = model_ret["population"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the atomic charge population using the lower-level interface."""
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            charge_spin=charge_spin,
        )
        model_predict = {}
        model_predict["population"] = model_ret["population"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict
