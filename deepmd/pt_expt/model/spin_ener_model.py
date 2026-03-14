# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from .spin_model import (
    SpinModel,
)


class SpinEnergyModel(SpinModel):
    """A spin model for energy."""

    model_type = "ener"

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
            "mask_mag": out_def_data["mask_mag"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
            output_def["force"].squeeze(-2)
            output_def["force_mag"] = deepcopy(out_def_data["energy_derv_r_mag"])
            output_def["force_mag"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
            output_def["atom_virial"].squeeze(-2)
        return output_def

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_common(
            coord,
            atype,
            spin,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["mask_mag"] = model_ret["mask_mag"]
        if self.backbone_model.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            model_predict["force_mag"] = model_ret["energy_derv_r_mag"].squeeze(-2)
        if self.backbone_model.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        return model_predict

    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["extended_mask_mag"] = model_ret["mask_mag"]
        if self.backbone_model.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            model_predict["extended_force_mag"] = model_ret[
                "energy_derv_r_mag"
            ].squeeze(-2)
        if self.backbone_model.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -2
                )
        return model_predict

    def forward_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
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
        extended_coord, extended_atype, extended_spin, nlist, mapping, fparam, aparam, do_atomic_virial
            Sample inputs with representative shapes (used for tracing).
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts
            ``(extended_coord, extended_atype, extended_spin, nlist, mapping, fparam, aparam)``
            and returns a dict with the same keys as ``forward_lower``.
        """
        traced = self.forward_common_lower_exportable(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            **make_fx_kwargs,
        )

        # Translate internal keys to forward_lower convention.
        # Capture model config at trace time via closures.
        do_grad_r = self.backbone_model.do_grad_r("energy")
        do_grad_c = self.backbone_model.do_grad_c("energy")

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            extended_spin: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            model_ret = traced(
                extended_coord,
                extended_atype,
                extended_spin,
                nlist,
                mapping,
                fparam,
                aparam,
            )
            model_predict: dict[str, torch.Tensor] = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            model_predict["extended_mask_mag"] = model_ret["mask_mag"]
            if do_grad_r:
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
                model_predict["extended_force_mag"] = model_ret[
                    "energy_derv_r_mag"
                ].squeeze(-2)
            if do_grad_c:
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-2)
            return model_predict

        return make_fx(fn, **make_fx_kwargs)(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping,
            fparam,
            aparam,
        )
