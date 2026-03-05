# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.dpmodel.model.make_hessian_model import (
    make_hessian_model,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
        self._hessian_enabled = False

    def enable_hessian(self) -> None:
        if self._hessian_enabled:
            return
        self.__class__ = make_hessian_model(type(self))
        self.hess_fitting_def = copy.deepcopy(
            super(type(self), self).atomic_output_def()
        )
        self.requires_hessian("energy")
        self._hessian_enabled = True

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
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        if self.atomic_output_def()["energy"].r_hessian:
            model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
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
        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
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
        if self.atomic_output_def()["energy"].r_hessian:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
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
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace ``forward_lower`` into an exportable module.

        Delegates to ``forward_common_lower_exportable`` for tracing,
        then translates the internal keys to the ``forward_lower``
        convention.

        Parameters
        ----------
        extended_coord, extended_atype, nlist, mapping, fparam, aparam, do_atomic_virial
            Sample inputs with representative shapes (used for tracing).
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts
            ``(extended_coord, extended_atype, nlist, mapping, fparam, aparam)``
            and returns a dict with the same keys as ``forward_lower``.
        """
        traced = self.forward_common_lower_exportable(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            **make_fx_kwargs,
        )

        # Translate internal keys to forward_lower convention.
        # Capture model config at trace time via closures.
        do_grad_r = self.do_grad_r("energy")
        do_grad_c = self.do_grad_c("energy")

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            model_ret = traced(
                extended_coord, extended_atype, nlist, mapping, fparam, aparam
            )
            model_predict: dict[str, torch.Tensor] = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if do_grad_r:
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if do_grad_c:
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-2)
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
            return model_predict

        return make_fx(fn, **make_fx_kwargs)(
            extended_coord, extended_atype, nlist, mapping, fparam, aparam
        )
